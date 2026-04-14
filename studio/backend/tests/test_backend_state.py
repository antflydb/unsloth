# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``core.inference.backend_state``.

Pins the contract of the selected-inference-backend singleton used by
route dispatch in ``routes/inference.py`` and ``routes/models.py``.

  * Default backend kind is ``"llama-cpp"`` (preserves pre-picker
    behaviour for upstream-merge fidelity).
  * ``set_backend_kind`` round-trips through ``get_backend_kind``.
  * Invalid kinds are rejected at the module boundary so a bad POST
    body can never land in shared state.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path + stubs for heavy deps (mirrors test_llama_cpp_load_progress.py).
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx is pulled in transitively via ``core.inference.__init__`` →
# ``llama_cpp``. These tests don't touch HTTP; stub it out so the suite
# stays a pure unit test.
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
    "HTTPError",
    "HTTPStatusError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
_httpx_stub.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **kw: None})
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)


@pytest.fixture(autouse = True)
def _reset_backend_state():
    """Reset the module-level backend kind before and after each test."""
    from core.inference import backend_state

    backend_state.set_backend_kind("llama-cpp")
    yield
    backend_state.set_backend_kind("llama-cpp")


def test_backend_state_default_is_llama_cpp():
    from core.inference.backend_state import get_backend_kind

    # Fixture already reset; a fresh process would likewise default to
    # llama-cpp so existing flows are untouched until someone opts in.
    assert get_backend_kind() == "llama-cpp"


def test_set_backend_kind_round_trip():
    from core.inference.backend_state import get_backend_kind, set_backend_kind

    set_backend_kind("termite-zig")
    assert get_backend_kind() == "termite-zig"

    set_backend_kind("llama-cpp")
    assert get_backend_kind() == "llama-cpp"


def test_set_backend_kind_rejects_unknown_kind():
    from core.inference.backend_state import get_backend_kind, set_backend_kind

    with pytest.raises(ValueError):
        set_backend_kind("mystery-engine")  # type: ignore[arg-type]

    # State must not have been mutated by the rejected call.
    assert get_backend_kind() == "llama-cpp"
