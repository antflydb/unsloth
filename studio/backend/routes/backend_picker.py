# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference-engine picker endpoints.

Kept in a dedicated module so ``routes/inference.py`` retains a minimal
diff against upstream Unsloth — the picker is a sibling concern, not a
modification of the existing load/chat flow. Dispatch conditionals in
other route modules read from ``core.inference.backend_state`` directly.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from loggers import get_logger

from core.inference.backend_state import (
    get_backend_kind,
    set_backend_kind,
)
from models.inference import SetBackendRequest, SetBackendResponse

logger = get_logger(__name__)
router = APIRouter()


# ── Backend accessors (indirected so tests can monkeypatch) ──
#
# These import lazily: test doubles can replace these module-level
# attributes before any request is served, so no real subprocess is
# spawned during unit testing.


def get_llama_cpp_backend():
    """Return the active llama.cpp backend singleton.

    Imported lazily so the heavy ``llama_cpp`` module is not loaded
    for tests that only care about picker dispatch.
    """
    from routes.inference import get_llama_cpp_backend as _impl

    return _impl()


def get_termite_backend():
    """Return the active termite-zig backend singleton."""
    from core.inference.backend_state import get_termite_backend as _impl

    return _impl()


def _unload_outgoing(previous_kind: str) -> Optional[str]:
    """Unload whatever model is loaded in the outgoing backend.

    Returns the unloaded model identifier (if any) so the caller can
    surface it in the response body for UI toasts / debugging.
    """
    if previous_kind == "llama-cpp":
        backend = get_llama_cpp_backend()
    elif previous_kind == "termite-zig":
        backend = get_termite_backend()
    else:
        return None

    if not backend.is_loaded:
        return None

    model_id = backend.model_identifier
    try:
        backend.unload_model()
    except Exception as exc:  # noqa: BLE001
        # Unload is best-effort: a stuck outgoing backend must not block
        # the picker flip, or the UI would be wedged when the user tries
        # to switch away from a backend whose subprocess died.
        logger.warning(
            "Failed to unload %s backend (%s): %s", previous_kind, model_id, exc
        )
        return None
    return model_id


@router.get("/backend")
async def get_backend() -> dict:
    """Return the currently selected inference backend kind."""
    return {"backend": get_backend_kind()}


@router.post("/backend", response_model = SetBackendResponse)
async def post_backend(req: SetBackendRequest) -> SetBackendResponse:
    """Flip the active inference backend, unloading any outgoing model."""
    previous = get_backend_kind()
    unloaded: Optional[str] = None

    if req.backend != previous:
        unloaded = _unload_outgoing(previous)
        set_backend_kind(req.backend)
        logger.info(
            "Switched inference backend: %s → %s (unloaded=%s)",
            previous,
            req.backend,
            unloaded,
        )

    return SetBackendResponse(
        backend = req.backend,
        previous_backend = previous,
        unloaded = unloaded,
    )
