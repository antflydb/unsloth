# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``core.inference.termite_bridge``.

The bridge downloads a GGUF via ``huggingface_hub`` (idempotent — HF's
cache does the heavy lifting) and creates symlinks into termite-zig's
rigid ``<models>/generators/<owner>/<name>/`` layout so the same file
on disk can be served by either llama.cpp or termite-zig. This
sidesteps the "download twice" problem without modifying termite.

Pure unit: the HF client is stubbed; the "cache" is a tmp_path.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# sys.path + stubs.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)


def _load_bridge_module():
    import importlib.util

    module_path = Path(_BACKEND_DIR) / "core" / "inference" / "termite_bridge.py"
    spec = importlib.util.spec_from_file_location(
        "core_inference_termite_bridge_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _install_fake_hf(monkeypatch, repo_files: list[str], local_paths: dict[str, Path]):
    """Install a fake ``huggingface_hub`` module.

    ``repo_files`` is what ``list_repo_files`` returns.
    ``local_paths`` maps filename -> on-disk path (the "cache hit" result).
    """
    hf = _types.ModuleType("huggingface_hub")
    hf.list_repo_files = lambda repo, token = None: list(repo_files)

    def _fake_download(repo_id, filename, token = None):
        # Raise if caller asks for a filename we didn't set up — this
        # catches drift between the bridge's file selection and the
        # fake repo contents.
        return str(local_paths[filename])

    hf.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    return hf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bridge_creates_symlink_for_single_gguf(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    # Simulate the HF cache layout: snapshots/<rev>/<file>.gguf.
    cache_file = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--gemma-3-4b-it-GGUF"
        / "snapshots"
        / "abc123"
        / "gemma-3-4b-it-Q4_K_M.gguf"
    )
    cache_file.parent.mkdir(parents = True)
    cache_file.write_bytes(b"FAKE GGUF")

    _install_fake_hf(
        monkeypatch,
        repo_files = [
            "README.md",
            "config.json",
            "gemma-3-4b-it-Q4_K_M.gguf",
            "gemma-3-4b-it-Q8_0.gguf",
        ],
        local_paths = {"gemma-3-4b-it-Q4_K_M.gguf": cache_file},
    )

    models_dir = tmp_path / "termite_models"
    result = bridge.bridge_gguf_to_termite(
        hf_repo = "unsloth/gemma-3-4b-it-GGUF",
        hf_variant = "Q4_K_M",
        models_dir = models_dir,
    )

    # Identifier termite will see is the HF repo id verbatim.
    assert result == "unsloth/gemma-3-4b-it-GGUF"

    expected_link = (
        models_dir
        / "generators"
        / "unsloth"
        / "gemma-3-4b-it-GGUF"
        / "gemma-3-4b-it-Q4_K_M.gguf"
    )
    assert expected_link.is_symlink(), f"expected symlink at {expected_link}"
    assert expected_link.resolve() == cache_file.resolve()


def test_bridge_symlinks_all_shards_for_split_gguf(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    snapshot_dir = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--BigModel-GGUF"
        / "snapshots"
        / "def456"
    )
    snapshot_dir.mkdir(parents = True)

    shards = [
        "BigModel-Q4_K_M-00001-of-00003.gguf",
        "BigModel-Q4_K_M-00002-of-00003.gguf",
        "BigModel-Q4_K_M-00003-of-00003.gguf",
    ]
    local_paths = {}
    for shard in shards:
        path = snapshot_dir / shard
        path.write_bytes(b"SHARD")
        local_paths[shard] = path

    _install_fake_hf(
        monkeypatch,
        repo_files = [
            "config.json",
            *shards,
            "BigModel-Q8_0-00001-of-00005.gguf",  # unrelated variant, different prefix
        ],
        local_paths = local_paths,
    )

    models_dir = tmp_path / "termite_models"
    result = bridge.bridge_gguf_to_termite(
        hf_repo = "unsloth/BigModel-GGUF",
        hf_variant = "Q4_K_M",
        models_dir = models_dir,
    )

    assert result == "unsloth/BigModel-GGUF"

    target_dir = models_dir / "generators" / "unsloth" / "BigModel-GGUF"
    for shard in shards:
        link = target_dir / shard
        assert link.is_symlink(), f"missing shard symlink {link}"
        assert link.resolve() == local_paths[shard].resolve()

    # Q8_0 shard must NOT be linked — the variant filter should have
    # excluded it. Otherwise termite would see a mismatched set.
    assert not (target_dir / "BigModel-Q8_0-00001-of-00005.gguf").exists()


def test_bridge_is_idempotent(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    cache_file = (
        tmp_path
        / "hf_cache"
        / "models--x--y"
        / "snapshots"
        / "rev"
        / "y-Q4_K_M.gguf"
    )
    cache_file.parent.mkdir(parents = True)
    cache_file.write_bytes(b"g")

    _install_fake_hf(
        monkeypatch,
        repo_files = ["y-Q4_K_M.gguf"],
        local_paths = {"y-Q4_K_M.gguf": cache_file},
    )

    models_dir = tmp_path / "termite_models"

    # First call creates the symlink.
    bridge.bridge_gguf_to_termite(
        hf_repo = "x/y", hf_variant = "Q4_K_M", models_dir = models_dir
    )
    link = models_dir / "generators" / "x" / "y" / "y-Q4_K_M.gguf"
    assert link.is_symlink()

    # Second call is a no-op — symlink still points at the same file.
    bridge.bridge_gguf_to_termite(
        hf_repo = "x/y", hf_variant = "Q4_K_M", models_dir = models_dir
    )
    assert link.is_symlink()
    assert link.resolve() == cache_file.resolve()


def test_bridge_fetches_tokenizer_from_base_repo_when_gguf_repo_lacks_it(
    tmp_path, monkeypatch
):
    """GGUF-only repos (e.g. ``unsloth/Foo-GGUF``) don't carry a
    ``tokenizer.json``; termite still needs one on disk. The bridge must
    fall back to the base repo (``unsloth/Foo``) for auxiliary files.
    """
    bridge = _load_bridge_module()

    gguf_cache = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--Qwen3.5-4B-GGUF"
        / "snapshots"
        / "g1"
        / "Qwen3.5-4B-Q4_K_M.gguf"
    )
    gguf_cache.parent.mkdir(parents = True)
    gguf_cache.write_bytes(b"gguf")

    tokenizer_cache = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--Qwen3.5-4B"
        / "snapshots"
        / "b1"
        / "tokenizer.json"
    )
    tokenizer_cache.parent.mkdir(parents = True)
    tokenizer_cache.write_text("{}")

    config_cache = tokenizer_cache.parent / "config.json"
    config_cache.write_text("{}")

    # Fake HF client: list_repo_files returns different contents per repo;
    # hf_hub_download raises FileNotFoundError for files that aren't in
    # that particular repo, so the bridge's fallback logic is exercised.
    import types

    hf = types.ModuleType("huggingface_hub")
    hf.list_repo_files = lambda repo, token = None: {
        "unsloth/Qwen3.5-4B-GGUF": [
            "README.md",
            "Qwen3.5-4B-Q4_K_M.gguf",
        ],
        "unsloth/Qwen3.5-4B": [
            "config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ],
    }[repo]

    gguf_repo_files = {"Qwen3.5-4B-Q4_K_M.gguf": gguf_cache}
    base_repo_files = {
        "tokenizer.json": tokenizer_cache,
        "config.json": config_cache,
    }

    def _fake_download(repo, filename, token = None):
        if repo == "unsloth/Qwen3.5-4B-GGUF":
            src = gguf_repo_files
        elif repo == "unsloth/Qwen3.5-4B":
            src = base_repo_files
        else:
            raise FileNotFoundError(repo)
        if filename not in src:
            # Mirror huggingface_hub's real behaviour.
            from huggingface_hub.utils import EntryNotFoundError  # type: ignore
            raise EntryNotFoundError(filename)
        return str(src[filename])

    hf.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    # Also stub the utils submodule so the bridge's error-type import
    # (if any) works. We don't actually raise EntryNotFoundError here —
    # we just need a generic exception to trigger the fallback.
    utils_mod = types.ModuleType("huggingface_hub.utils")

    class _ENF(Exception):
        pass

    utils_mod.EntryNotFoundError = _ENF
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils_mod)
    # Re-wire the download function to use the stubbed exception type.
    def _fake_download2(repo, filename, token = None):
        srcs = {
            "unsloth/Qwen3.5-4B-GGUF": gguf_repo_files,
            "unsloth/Qwen3.5-4B": base_repo_files,
        }
        src = srcs.get(repo)
        if src is None or filename not in src:
            raise _ENF(filename)
        return str(src[filename])

    hf.hf_hub_download = _fake_download2

    models_dir = tmp_path / "termite_models"
    result = bridge.bridge_gguf_to_termite(
        hf_repo = "unsloth/Qwen3.5-4B-GGUF",
        hf_variant = "Q4_K_M",
        models_dir = models_dir,
    )

    assert result == "unsloth/Qwen3.5-4B-GGUF"

    model_dir = models_dir / "generators" / "unsloth" / "Qwen3.5-4B-GGUF"
    assert (model_dir / "Qwen3.5-4B-Q4_K_M.gguf").is_symlink()
    # This is the key assertion — the tokenizer must be bridged from
    # the base repo so termite's load doesn't throw NoTokenizerFound.
    assert (model_dir / "tokenizer.json").is_symlink()
    assert (model_dir / "tokenizer.json").resolve() == tokenizer_cache.resolve()
    assert (model_dir / "config.json").is_symlink()


def test_bridge_raises_when_variant_not_in_repo(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    _install_fake_hf(
        monkeypatch,
        repo_files = ["only-Q4_K_M.gguf", "only-Q8_0.gguf"],
        local_paths = {},
    )

    with pytest.raises(RuntimeError, match = "variant"):
        bridge.bridge_gguf_to_termite(
            hf_repo = "x/y",
            hf_variant = "IQ2_XXS",
            models_dir = tmp_path / "termite_models",
        )


def test_bridge_hf_cache_to_termite_exposes_cached_gguf_repo(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    snapshot = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--Gemma-GGUF"
        / "snapshots"
        / "rev"
    )
    snapshot.mkdir(parents = True)
    gguf = snapshot / "Gemma-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_text("{}")
    ignored = snapshot / "README.md"
    ignored.write_text("ignore")

    hf = _types.ModuleType("huggingface_hub")
    hf.scan_cache_dir = lambda cache_dir = None: SimpleNamespace(
        repos = [
            SimpleNamespace(
                repo_type = "model",
                repo_id = "unsloth/Gemma-GGUF",
                revisions = [
                    SimpleNamespace(
                        files = [
                            SimpleNamespace(
                                file_name = "Gemma-Q4_K_M.gguf",
                                file_path = gguf,
                            ),
                            SimpleNamespace(
                                file_name = "tokenizer.json",
                                file_path = tokenizer,
                            ),
                            SimpleNamespace(
                                file_name = "README.md",
                                file_path = ignored,
                            ),
                        ]
                    )
                ],
            )
        ]
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)

    models_dir = tmp_path / "termite_models"
    bridged = bridge.bridge_hf_cache_to_termite(models_dir = models_dir)

    assert bridged == ["unsloth/Gemma-GGUF"]
    dest = models_dir / "generators" / "unsloth" / "Gemma-GGUF"
    assert (dest / "Gemma-Q4_K_M.gguf").is_symlink()
    assert (dest / "Gemma-Q4_K_M.gguf").resolve() == gguf.resolve()
    assert (dest / "tokenizer.json").is_symlink()
    assert (dest / "tokenizer.json").resolve() == tokenizer.resolve()
    assert not (dest / "README.md").exists()


def test_bridge_hf_cache_to_termite_ignores_non_gguf_repos(tmp_path, monkeypatch):
    bridge = _load_bridge_module()

    snapshot = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--Gemma"
        / "snapshots"
        / "rev"
    )
    snapshot.mkdir(parents = True)
    weights = snapshot / "model.safetensors"
    weights.write_bytes(b"weights")

    hf = _types.ModuleType("huggingface_hub")
    hf.scan_cache_dir = lambda cache_dir = None: SimpleNamespace(
        repos = [
            SimpleNamespace(
                repo_type = "model",
                repo_id = "unsloth/Gemma",
                revisions = [
                    SimpleNamespace(
                        files = [
                            SimpleNamespace(
                                file_name = "model.safetensors",
                                file_path = weights,
                            )
                        ]
                    )
                ],
            )
        ]
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)

    models_dir = tmp_path / "termite_models"
    assert bridge.bridge_hf_cache_to_termite(models_dir = models_dir) == []
    assert not (models_dir / "generators").exists()


def test_bridge_hf_cache_to_termite_fetches_missing_aux_from_base_repo(
    tmp_path,
    monkeypatch,
):
    bridge = _load_bridge_module()

    snapshot = (
        tmp_path
        / "hf_cache"
        / "models--unsloth--Gemma-GGUF"
        / "snapshots"
        / "rev"
    )
    snapshot.mkdir(parents = True)
    gguf = snapshot / "Gemma-Q4_K_M.gguf"
    gguf.write_bytes(b"gguf")

    base_snapshot = tmp_path / "hf_cache" / "models--unsloth--Gemma" / "snapshots" / "rev"
    base_snapshot.mkdir(parents = True)
    tokenizer = base_snapshot / "tokenizer.json"
    tokenizer.write_text("{}")
    config = base_snapshot / "config.json"
    config.write_text("{}")

    hf = _types.ModuleType("huggingface_hub")
    hf.scan_cache_dir = lambda cache_dir = None: SimpleNamespace(
        repos = [
            SimpleNamespace(
                repo_type = "model",
                repo_id = "unsloth/Gemma-GGUF",
                revisions = [
                    SimpleNamespace(
                        files = [
                            SimpleNamespace(
                                file_name = "Gemma-Q4_K_M.gguf",
                                file_path = gguf,
                            ),
                        ]
                    )
                ],
            )
        ]
    )

    def _fake_download(repo, filename, token = None):
        if repo == "unsloth/Gemma" and filename == "tokenizer.json":
            return str(tokenizer)
        if repo == "unsloth/Gemma" and filename == "config.json":
            return str(config)
        raise FileNotFoundError(filename)

    hf.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)

    models_dir = tmp_path / "termite_models"
    bridged = bridge.bridge_hf_cache_to_termite(models_dir = models_dir)

    assert bridged == ["unsloth/Gemma-GGUF"]
    dest = models_dir / "generators" / "unsloth" / "Gemma-GGUF"
    assert (dest / "Gemma-Q4_K_M.gguf").is_symlink()
    assert (dest / "tokenizer.json").resolve() == tokenizer.resolve()
    assert (dest / "config.json").resolve() == config.resolve()
