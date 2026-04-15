# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the termite-zig chat streaming helper.

``routes/termite_chat.py`` holds the full request→SSE translation for
the termite-zig backend so ``routes/inference.py`` only needs a tiny
early-return when the picker is flipped to termite. The helper itself
is pure and unit-testable without spinning up FastAPI.
"""

from __future__ import annotations

import asyncio
import json
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


def _load_helper():
    import importlib.util

    module_path = Path(_BACKEND_DIR) / "routes" / "termite_chat.py"
    spec = importlib.util.spec_from_file_location(
        "routes_termite_chat_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTermite:
    def __init__(self, cumulative_pieces):
        self._pieces = cumulative_pieces
        self.is_loaded = True
        self.model_identifier = "google/gemma-3-1b-it"

    def generate_chat_completion(self, *, messages, cancel_event = None, **_):
        # Yield cumulative strings like the real backend.
        for text in self._pieces:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield text


class _FakePayload:
    """Stand-in for ``ChatCompletionRequest``; only the fields used by the
    helper are populated."""

    def __init__(self):
        self.stream = True
        self.model = "google/gemma-3-1b-it"
        self.messages = [{"role": "user", "content": "hi"}]
        self.temperature = 0.7
        self.top_p = 0.95
        self.top_k = 20
        self.min_p = 0.01
        self.max_tokens = 16
        self.repetition_penalty = 1.0
        self.presence_penalty = 0.0
        self.enable_thinking = None
        self.audio_base64 = None
        self.image_base64 = None
        self.image_url = None


class _FakeRequest:
    """Stand-in for FastAPI's ``Request``; we only need ``is_disconnected``."""

    async def is_disconnected(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _collect_sse(lines) -> list[dict]:
    """Parse a list of SSE lines into structured events."""
    events: list[dict] = []
    for raw in lines:
        raw = raw.strip()
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if payload == "[DONE]":
            events.append({"_done": True})
            continue
        events.append(json.loads(payload))
    return events


def test_stream_emits_role_then_deltas_then_done():
    helper = _load_helper()
    fake = _FakeTermite(["Hel", "Hello", "Hello, world"])

    async def run():
        gen = helper.stream_termite_chat(
            payload = _FakePayload(),
            request = _FakeRequest(),
            termite_backend = fake,
        )
        return [chunk async for chunk in gen]

    chunks = asyncio.get_event_loop().run_until_complete(run()) \
        if sys.version_info < (3, 10) else asyncio.run(run())

    events = _collect_sse(chunks)

    # Role chunk first.
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"

    # Then one delta per incremental slice.
    deltas = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if not e.get("_done") and "choices" in e and "content" in e["choices"][0].get("delta", {})
    ]
    assert deltas == ["Hel", "lo", ", world"]

    # [DONE] sentinel last.
    assert events[-1].get("_done") is True


def test_stream_stops_on_disconnect():
    helper = _load_helper()
    fake = _FakeTermite(["a", "ab", "abc", "abcd"])

    class _Disc(_FakeRequest):
        def __init__(self):
            self._calls = 0

        async def is_disconnected(self) -> bool:
            self._calls += 1
            # Disconnect after the first successful chunk has been
            # emitted. We expect the helper to short-circuit and not
            # yield any further deltas.
            return self._calls >= 3

    async def run():
        gen = helper.stream_termite_chat(
            payload = _FakePayload(),
            request = _Disc(),
            termite_backend = fake,
        )
        return [chunk async for chunk in gen]

    chunks = asyncio.run(run())
    events = _collect_sse(chunks)

    contents = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if not e.get("_done") and "choices" in e and "content" in e["choices"][0].get("delta", {})
    ]
    # At least one content chunk should have been emitted, but the
    # stream must end before emitting all four.
    assert 0 < len(contents) < 4


def test_stream_raises_when_no_model_selected():
    helper = _load_helper()

    class _Unloaded:
        is_loaded = False
        model_identifier = None

    async def run():
        gen = helper.stream_termite_chat(
            payload = _FakePayload(),
            request = _FakeRequest(),
            termite_backend = _Unloaded(),
        )
        return [chunk async for chunk in gen]

    with pytest.raises(RuntimeError, match = "no model"):
        asyncio.run(run())
