# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Selected-backend state for the inference-engine picker.

Studio defaults to the ``llama-cpp`` backend (``LlamaCppBackend``),
preserving every existing chat flow. When the picker flips to
``termite-zig``, routes dispatch to ``TermiteZigBackend`` at four
well-known seams in ``routes/inference.py`` and ``routes/models.py``.

State lives in this module-level singleton (not a class) so route
handlers can import and check it without pulling in the full backend
factory graph, and so upstream merges see a minimal diff — all
dispatch conditionals read a single ``get_backend_kind()`` call.

Intentionally tiny: no cross-backend locking, no auto-spawning — that
belongs on the backend class itself.
"""

from __future__ import annotations

import threading
from typing import Literal, get_args

BackendKind = Literal["llama-cpp", "termite-zig"]

_VALID_KINDS: frozenset[str] = frozenset(get_args(BackendKind))

_lock = threading.Lock()
_state: dict[str, BackendKind] = {"kind": "llama-cpp"}


def get_backend_kind() -> BackendKind:
    """Return the currently selected inference backend kind."""
    with _lock:
        return _state["kind"]


def set_backend_kind(kind: BackendKind) -> None:
    """Set the selected inference backend kind.

    Raises ``ValueError`` if ``kind`` isn't one of the literal members
    of ``BackendKind``. Validation happens here so a bad POST body
    can't land in shared state, even if the schema layer is bypassed.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown backend kind: {kind!r}; expected one of {sorted(_VALID_KINDS)}"
        )
    with _lock:
        _state["kind"] = kind
