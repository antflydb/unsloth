# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``POST /api/inference/backend`` and ``GET /api/inference/backend``.

The picker endpoints flip the ``backend_state`` singleton. The hard
requirement is that switching **away** from a currently-loaded backend
must unload the outgoing model first — otherwise we'd hold a stale GGUF
in page cache for the whole session while the user is trying to run
termite-zig, or the other way round.

Fake backends are sufficient here: we're verifying the dispatcher's
state transitions, not subprocess plumbing.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path + stubs for heavy deps.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)


def _load_backend_picker_module():
    """Import ``routes.backend_picker`` without running ``routes/__init__.py``.

    The package init eagerly imports every router, dragging in the full
    training stack (matplotlib, transformers, …). This test only needs
    the picker module itself.
    """
    import importlib.util

    module_path = Path(_BACKEND_DIR) / "routes" / "backend_picker.py"
    spec = importlib.util.spec_from_file_location(
        "routes_backend_picker_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _FakeLlamaCpp:
    def __init__(self, loaded = False, model_id = None):
        self._loaded = loaded
        self._model_id = model_id
        self.unload_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_identifier(self):
        return self._model_id

    def unload_model(self):
        self.unload_count += 1
        self._loaded = False
        self._model_id = None
        return True


class _FakeTermite:
    def __init__(self, loaded = False, model_id = None):
        self._loaded = loaded
        self._model_id = model_id
        self.unload_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_identifier(self):
        return self._model_id

    def unload_model(self):
        self.unload_count += 1
        self._loaded = False
        self._model_id = None
        return True


# ---------------------------------------------------------------------------
# Fixture: mini FastAPI app wrapping just the picker endpoints.
# ---------------------------------------------------------------------------


@pytest.fixture
def picker_client(monkeypatch):
    """Return ``(client, fakes)``.

    ``fakes`` is an object exposing ``.llama`` and ``.termite`` so tests
    can mutate state before the request and assert against afterwards.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Reset backend_state each test.
    from core.inference import backend_state
    backend_state.set_backend_kind("llama-cpp")

    llama = _FakeLlamaCpp()
    termite = _FakeTermite()

    # Import the routes module lazily so stubs above are in effect.
    picker_module = _load_backend_picker_module()

    monkeypatch.setattr(picker_module, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(picker_module, "get_termite_backend", lambda: termite)

    app = FastAPI()
    app.include_router(picker_module.router, prefix = "/api/inference")

    class _Fakes:
        pass

    fakes = _Fakes()
    fakes.llama = llama
    fakes.termite = termite

    with TestClient(app) as client:
        yield client, fakes

    backend_state.set_backend_kind("llama-cpp")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_backend_defaults_to_llama_cpp(picker_client):
    client, _fakes = picker_client

    resp = client.get("/api/inference/backend")

    assert resp.status_code == 200
    assert resp.json() == {"backend": "llama-cpp"}


def test_post_backend_switches_and_unloads_outgoing_llama_cpp(picker_client):
    client, fakes = picker_client
    fakes.llama._loaded = True
    fakes.llama._model_id = "unsloth/gemma-3-4b-it-GGUF"

    resp = client.post(
        "/api/inference/backend",
        json = {"backend": "termite-zig"},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "backend": "termite-zig",
        "previous_backend": "llama-cpp",
        "unloaded": "unsloth/gemma-3-4b-it-GGUF",
    }
    assert fakes.llama.unload_count == 1

    # Confirm the new kind is persisted.
    follow_up = client.get("/api/inference/backend")
    assert follow_up.json() == {"backend": "termite-zig"}


def test_post_backend_unloads_outgoing_termite(picker_client):
    client, fakes = picker_client

    from core.inference import backend_state
    backend_state.set_backend_kind("termite-zig")
    fakes.termite._loaded = True
    fakes.termite._model_id = "google/gemma-3-1b-it"

    resp = client.post(
        "/api/inference/backend",
        json = {"backend": "llama-cpp"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "llama-cpp"
    assert body["previous_backend"] == "termite-zig"
    assert body["unloaded"] == "google/gemma-3-1b-it"
    assert fakes.termite.unload_count == 1


def test_post_backend_no_unload_when_same_kind(picker_client):
    client, fakes = picker_client
    fakes.llama._loaded = True
    fakes.llama._model_id = "unsloth/gemma-3-4b-it-GGUF"

    resp = client.post(
        "/api/inference/backend",
        json = {"backend": "llama-cpp"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "llama-cpp"
    assert body["previous_backend"] == "llama-cpp"
    assert body["unloaded"] is None
    # The outgoing backend is the same as the incoming — don't unload.
    assert fakes.llama.unload_count == 0


def test_post_backend_rejects_unknown_kind(picker_client):
    client, _fakes = picker_client

    resp = client.post(
        "/api/inference/backend",
        json = {"backend": "mystery-engine"},
    )

    # FastAPI / Pydantic rejects unknown Literal values with 422.
    assert resp.status_code == 422
