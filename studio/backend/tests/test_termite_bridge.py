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
