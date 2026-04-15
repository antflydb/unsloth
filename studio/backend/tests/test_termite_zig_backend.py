# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``TermiteZigBackend`` — the picker's non-default backend.

Pure unit tests. No real subprocess, no real HTTP. Verifies:

  * Binary discovery honours ``TERMITE_BIN`` → repo-relative path →
    ``~/.unsloth/termite/bin/termite`` → ``$PATH``.
  * ``TERMITE_ZIG_URL`` short-circuits spawn entirely (the user points
    at an externally-managed termite instance, e.g. one they launched
    under a debugger).
  * ``list_models()`` proxies ``GET /ml/v1/models`` and maps the
    OpenAI-style entries into the ``LocalModelInfo`` shape the rest
    of Studio expects.
  * ``generate_chat_completion()`` parses SSE chunks from the termite
    chat endpoint and yields cumulative text, matching the convention
    ``LlamaCppBackend.generate_chat_completion`` uses.
"""

from __future__ import annotations

import os
import sys
import types as _types
from pathlib import Path
from typing import Any, Callable, Optional

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


def _load_termite_zig_module():
    """Import the backend module directly, avoiding ``routes/__init__.py``."""
    import importlib.util

    module_path = Path(_BACKEND_DIR) / "core" / "inference" / "termite_zig.py"
    spec = importlib.util.spec_from_file_location(
        "core_inference_termite_zig_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Binary discovery (test 4)
# ---------------------------------------------------------------------------


def test_find_termite_binary_from_env(tmp_path, monkeypatch):
    termite_zig = _load_termite_zig_module()
    fake_bin = tmp_path / "termite"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)

    monkeypatch.setenv("TERMITE_BIN", str(fake_bin))

    assert termite_zig.TermiteZigBackend._find_termite_binary() == str(fake_bin)


def test_find_termite_binary_falls_back_to_repo_path(tmp_path, monkeypatch):
    """When TERMITE_BIN is unset, the repo-relative dev build wins."""
    termite_zig = _load_termite_zig_module()

    # Simulate a clean env with no TERMITE_BIN.
    monkeypatch.delenv("TERMITE_BIN", raising = False)

    # Create a fake repo layout: repo_root/termite-zig/zig-out/bin/termite
    fake_repo = tmp_path / "repo"
    fake_bin = fake_repo / "termite-zig" / "zig-out" / "bin" / "termite"
    fake_bin.parent.mkdir(parents = True)
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)

    # Point the backend's REPO_ROOT at our fake layout.
    monkeypatch.setattr(termite_zig, "_REPO_ROOT", str(fake_repo))

    assert termite_zig.TermiteZigBackend._find_termite_binary() == str(fake_bin)


def test_find_termite_binary_returns_none_when_nothing_found(tmp_path, monkeypatch):
    termite_zig = _load_termite_zig_module()

    monkeypatch.delenv("TERMITE_BIN", raising = False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.unsloth/termite/bin/termite
    monkeypatch.setattr(termite_zig, "_REPO_ROOT", str(tmp_path / "nope"))
    # Scrub PATH so ``shutil.which`` can't find a system termite.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert termite_zig.TermiteZigBackend._find_termite_binary() is None


# ---------------------------------------------------------------------------
# URL override (test 5)
# ---------------------------------------------------------------------------


def test_ensure_running_short_circuits_on_termite_zig_url(monkeypatch):
    """With TERMITE_ZIG_URL set, no subprocess is spawned.

    The override is the debug / bring-your-own-termite path. The
    backend should probe readiness at the supplied URL and never
    shell out to Popen.
    """
    termite_zig = _load_termite_zig_module()

    monkeypatch.setenv("TERMITE_ZIG_URL", "http://127.0.0.1:9999")

    popen_calls: list[tuple] = []

    def _fake_popen(*args, **kwargs):  # pragma: no cover - asserted below
        popen_calls.append((args, kwargs))
        raise AssertionError("Popen should not be called when URL override is set")

    monkeypatch.setattr(termite_zig.subprocess, "Popen", _fake_popen)

    # Fake httpx readiness probe: respond 200 immediately.
    class _FakeResp:
        status_code = 200

        def json(self):
            return {"status": "ok"}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        termite_zig.httpx,
        "get",
        lambda url, timeout = None: _FakeResp(),
    )

    backend = termite_zig.TermiteZigBackend()
    backend._ensure_running()

    assert popen_calls == []
    assert backend.base_url == "http://127.0.0.1:9999"
    assert backend._process is None


def test_ensure_running_url_override_retries_until_ready(monkeypatch):
    """Probes ``/readyz`` with retry; fails after a reasonable timeout
    rather than hanging if the externally-managed termite never comes up.
    """
    termite_zig = _load_termite_zig_module()

    monkeypatch.setenv("TERMITE_ZIG_URL", "http://127.0.0.1:9999")

    class _ConnectError(Exception):
        pass

    termite_zig.httpx.ConnectError = _ConnectError  # type: ignore[attr-defined]

    def _always_connect_error(url, timeout = None):
        raise _ConnectError("refused")

    monkeypatch.setattr(termite_zig.httpx, "get", _always_connect_error)
    # Collapse sleep so the retry loop exits immediately.
    monkeypatch.setattr(termite_zig.time, "sleep", lambda _: None)
    monkeypatch.setattr(termite_zig, "_READY_MAX_WAIT_SECONDS", 0.01)

    backend = termite_zig.TermiteZigBackend()
    with pytest.raises(RuntimeError, match = "not ready"):
        backend._ensure_running()


# ---------------------------------------------------------------------------
# list_models (test 6)
# ---------------------------------------------------------------------------


def test_list_models_maps_openai_shape(monkeypatch):
    termite_zig = _load_termite_zig_module()

    monkeypatch.setenv("TERMITE_ZIG_URL", "http://127.0.0.1:9999")

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "object": "list",
                "data": [
                    {
                        "id": "google/gemma-3-1b-it",
                        "object": "model",
                        "owned_by": "google",
                    },
                    {
                        "id": "bge-small-en-v1.5",
                        "object": "model",
                        "owned_by": "bge",
                    },
                ],
            }

        def raise_for_status(self):
            return None

    # Ready probe + subsequent list call both use httpx.get.
    monkeypatch.setattr(termite_zig.httpx, "get", lambda url, timeout = None: _FakeResp())

    backend = termite_zig.TermiteZigBackend()
    backend._ensure_running()
    models = backend.list_models()

    ids = [m["id"] for m in models]
    assert "google/gemma-3-1b-it" in ids
    assert "bge-small-en-v1.5" in ids
    # The picker v1 only cares about text chat — but termite surfaces
    # embedders/rerankers in the same list. We keep them and let the
    # frontend filter; tested elsewhere.
    for m in models:
        assert "id" in m


# ---------------------------------------------------------------------------
# generate_chat_completion (test 7)
# ---------------------------------------------------------------------------


def test_generate_chat_completion_streams_cumulative_text(monkeypatch):
    """Fake an SSE stream of OpenAI chunks; assert cumulative yields."""
    termite_zig = _load_termite_zig_module()

    monkeypatch.setenv("TERMITE_ZIG_URL", "http://127.0.0.1:9999")

    sse_lines = [
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":", world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    class _FakeStreamResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self):
            for line in sse_lines:
                yield line.decode("utf-8").rstrip("\n")

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def stream(self, method, url, json = None, headers = None):
            return _FakeCtx(_FakeStreamResponse())

    class _FakeCtx:
        def __init__(self, resp):
            self._resp = resp

        def __enter__(self):
            return self._resp

        def __exit__(self, *a):
            return None

    monkeypatch.setattr(termite_zig.httpx, "Client", _FakeClient)
    # Ready probe: single 200.
    class _Ready:
        status_code = 200
        def json(self): return {"status": "ok"}
        def raise_for_status(self): return None
    monkeypatch.setattr(termite_zig.httpx, "get", lambda url, timeout = None: _Ready())

    backend = termite_zig.TermiteZigBackend()
    backend._ensure_running()
    backend._model_identifier = "google/gemma-3-1b-it"

    chunks = list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            max_tokens = 16,
        )
    )

    # Llama.cpp convention: each yield is the CUMULATIVE text so far,
    # so the last yield equals the full response. (Matches how
    # chat-adapter.ts treats the stream.)
    text_yields = [c for c in chunks if isinstance(c, str)]
    assert text_yields, f"expected text yields; got {chunks!r}"
    assert text_yields[-1] == "Hello, world"
    # Monotonic growth.
    for a, b in zip(text_yields, text_yields[1:]):
        assert b.startswith(a)
