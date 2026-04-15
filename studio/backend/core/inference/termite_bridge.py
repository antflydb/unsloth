# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
HF-cache → termite-zig symlink bridge.

Studio already downloads GGUFs into the HuggingFace cache
(``~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<rev>/``).
termite-zig has a rigid layout it won't deviate from:

    <models>/generators/<owner>/<name>/<filename>.gguf

Rather than force a second download through ``termite pull`` (bad UX,
doubles disk use, duplicates HF hub rate limit), we symlink the
HF-cached file into termite's expected layout. termite's scanner picks
it up as if it were a native download — no termite code changes
required.

The bridge is idempotent: safe to call on every load.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# GGUF shards look like ``foo-00001-of-00003.gguf``.
_SHARD_FULL_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")

# Tokenizer / config files termite needs to load a GGUF. GGUFs do embed
# most of this metadata, but current termite-zig (v0.1.0) still reads
# ``tokenizer.json`` off disk and fails with ``NoTokenizerFound`` if
# it's missing. We symlink what the HF repo has; missing files are
# silently skipped so the bridge stays a best-effort helper.
_TOKENIZER_AUXILIARY_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _candidate_base_repos(gguf_repo: str) -> list[str]:
    """Guess the non-GGUF sibling repo that holds tokenizer/config files.

    Convention: ``<owner>/<name>-GGUF`` tends to be the quantized build of
    ``<owner>/<name>``. Studio's own models follow this pattern. We try
    the canonical suffix strips first, then fall back to the GGUF repo
    itself so the caller can also look in-place.
    """
    candidates: list[str] = []
    for suffix in ("-GGUF", "-gguf"):
        if gguf_repo.endswith(suffix):
            candidates.append(gguf_repo[: -len(suffix)])
    candidates.append(gguf_repo)
    # Preserve order while deduplicating.
    seen: set[str] = set()
    out: list[str] = []
    for r in candidates:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _try_download_aux(
    repo: str, filename: str, token: Optional[str]
) -> Optional[Path]:
    """Attempt to download ``filename`` from ``repo``. None on 404 / error."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, filename, token = token))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "termite bridge: no %s in %s (%s)", filename, repo, type(exc).__name__
        )
        return None


def termite_models_dir() -> Path:
    """Resolve termite's models dir.

    Honours ``TERMITE_MODELS_DIR`` (matches termite's own ``--models``
    flag) and otherwise defaults to ``~/.termite/models``.
    """
    override = os.environ.get("TERMITE_MODELS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".termite" / "models"


def _symlink_idempotent(target: Path, link_path: Path) -> None:
    """Create or refresh a symlink at ``link_path`` pointing to ``target``.

    Idempotent semantics:
      * Correct existing symlink → no-op.
      * Broken / wrongly-pointed symlink → replaced.
      * Non-symlink file already there → left alone (caller loses the
        race with e.g. a ``termite pull``, but that's strictly safer
        than overwriting a real download).
    """
    target_resolved = target.resolve()

    if link_path.is_symlink():
        try:
            if link_path.resolve(strict = True) == target_resolved:
                return
        except (OSError, RuntimeError):
            pass
        link_path.unlink()
    elif link_path.exists():
        logger.info(
            "termite bridge: %s exists and is not a symlink; not overwriting",
            link_path,
        )
        return

    link_path.parent.mkdir(parents = True, exist_ok = True)
    link_path.symlink_to(target_resolved)


def _pick_variant_match(
    gguf_files: list[str], hf_variant: Optional[str]
) -> str:
    """Pick a single GGUF filename matching the variant.

    Mirrors the matcher in ``LlamaCppBackend._download_gguf`` (word-
    boundary on the variant token, lowercased) so both backends resolve
    the same repo+variant to the same file.
    """
    if not hf_variant:
        return sorted(gguf_files)[0]
    boundary = re.compile(
        r"(?<![a-zA-Z0-9])" + re.escape(hf_variant.lower()) + r"(?![a-zA-Z0-9])"
    )
    matches = sorted(f for f in gguf_files if boundary.search(f.lower()))
    if not matches:
        raise RuntimeError(
            f"No GGUF matches variant {hf_variant!r} in repo (candidates: "
            f"{gguf_files[:5]}...)"
        )
    return matches[0]


def _collect_shards(main: str, gguf_files: list[str]) -> list[str]:
    """Return ``main`` plus any sibling shards that belong to the same split."""
    m = _SHARD_FULL_RE.match(main)
    if not m:
        return [main]
    prefix = m.group(1)
    total = m.group(3)
    sibling_pat = re.compile(
        r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(total) + r"\.gguf$"
    )
    return sorted(f for f in gguf_files if sibling_pat.match(f))


def bridge_gguf_to_termite(
    *,
    hf_repo: str,
    hf_variant: Optional[str],
    hf_token: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> str:
    """Download (if needed) and symlink a GGUF into termite-zig's layout.

    Reuses Studio's existing HF cache via ``huggingface_hub`` — no
    duplicate download, no second HTTP call if the file is already
    cached locally.

    Returns the identifier termite-zig will surface the model as
    (``<owner>/<name>`` — the HF repo id verbatim, since termite's
    2-level layout happens to match HF's owner/name convention).
    """
    # Import lazily so test suites that don't touch this module pay no
    # huggingface_hub import cost.
    from huggingface_hub import hf_hub_download, list_repo_files

    all_files = list_repo_files(hf_repo, token = hf_token)
    gguf_files = [f for f in all_files if f.endswith(".gguf")]
    if not gguf_files:
        raise RuntimeError(f"No GGUF files in repo {hf_repo!r}")

    main = _pick_variant_match(gguf_files, hf_variant)
    targets = _collect_shards(main, gguf_files)

    local_paths: list[Path] = []
    for filename in targets:
        path_str = hf_hub_download(hf_repo, filename, token = hf_token)
        local_paths.append(Path(path_str))

    dest_dir = (models_dir or termite_models_dir()) / "generators" / hf_repo
    for real_path in local_paths:
        _symlink_idempotent(real_path, dest_dir / real_path.name)

    # ── Tokenizer / config bridge ──────────────────────────────────
    # termite-zig (v0.1.0) fails with ``NoTokenizerFound`` on chat
    # completion if there's no on-disk tokenizer.json next to the
    # GGUF — even though GGUFs carry an embedded tokenizer. Work
    # around by fetching the companion files from the repo itself
    # or the likely base repo (``<name>-GGUF`` → ``<name>``).
    aux_count = 0
    for filename in _TOKENIZER_AUXILIARY_FILES:
        for candidate in _candidate_base_repos(hf_repo):
            aux_path = _try_download_aux(candidate, filename, hf_token)
            if aux_path is not None:
                _symlink_idempotent(aux_path, dest_dir / filename)
                aux_count += 1
                break  # stop on first repo that has this file

    logger.info(
        "termite bridge: %s variant=%s → %d GGUF + %d aux symlinks under %s",
        hf_repo,
        hf_variant,
        len(local_paths),
        aux_count,
        dest_dir,
    )
    # termite's discovery enumerates ``generators/<owner>/<name>`` and
    # exposes that 2-level key as the model id.
    return hf_repo
