# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level termite-zig integration tests for ``routes/inference.py``.

These sit one layer above the pure-unit helper/backend tests:

* ``POST /api/inference/load`` in termite mode should bridge GGUFs via the
  normal Studio cache path and then hand the resolved identifier to the
  termite backend. This protects the "no manual termite pull required" UX.
* ``POST /v1/chat/completions`` in termite mode should stream non-empty
  assistant deltas through the real route wiring, not just through the helper.
* Non-streaming clients should get a standard ``chat.completion`` payload.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace

from fastapi import Request


# ---------------------------------------------------------------------------
# sys.path + light stubs for heavy deps.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)


def _load_route_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name,
        Path(_BACKEND_DIR) / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _sse_payloads(chunks: list[str]) -> list[dict]:
    events: list[dict] = []
    for raw in chunks:
        for line in raw.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                events.append({"_done": True})
            else:
                events.append(json.loads(payload))
    return events


class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


class _FakeTermite:
    def __init__(self, cumulative_pieces: list[str] | None = None):
        self._pieces = cumulative_pieces or []
        self._loaded = True
        self._model_id = "unsloth/Llama-3.2-1B-Instruct-GGUF"
        self.load_calls: list[dict] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_identifier(self) -> str | None:
        return self._model_id

    def load_model(self, **kwargs):
        self.load_calls.append(dict(kwargs))
        self._loaded = True
        self._model_id = kwargs.get("model_identifier")
        return True

    def generate_chat_completion(self, *, messages, cancel_event = None, **_):
        for text in self._pieces:
            if cancel_event is not None and cancel_event.is_set():
                return
            yield text


class _ExplodingLlama:
    @property
    def is_loaded(self) -> bool:
        raise AssertionError("llama.cpp backend should not be consulted in termite mode")


async def _collect_streaming_response(response) -> list[str]:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return chunks


def test_load_route_in_termite_mode_bridges_gguf_and_passes_identifier(monkeypatch):
    inference_route = _load_route_module(
        "inference_route_module_for_termite_bridge_test",
        "routes/inference.py",
    )

    from models.inference import LoadRequest

    request = LoadRequest(
        model_path = "unsloth/Llama-3.2-1B-Instruct-GGUF",
        gguf_variant = "Q4_K_M",
        hf_token = "hf_test",
        max_seq_length = 8192,
    )
    config = SimpleNamespace(
        is_gguf = True,
        gguf_hf_repo = "unsloth/Llama-3.2-1B-Instruct-GGUF",
        gguf_variant = "Q4_K_M",
    )
    fake_termite = _FakeTermite()
    bridge_calls: list[dict] = []

    bridge_mod = _types.ModuleType("core.inference.termite_bridge")

    def _fake_bridge(**kwargs):
        bridge_calls.append(dict(kwargs))
        return "unsloth/Llama-3.2-1B-Instruct-GGUF"

    bridge_mod.bridge_gguf_to_termite = _fake_bridge
    monkeypatch.setitem(sys.modules, "core.inference.termite_bridge", bridge_mod)
    monkeypatch.setattr(inference_route, "get_backend_kind", lambda: "termite-zig")
    monkeypatch.setattr(inference_route, "get_termite_backend", lambda: fake_termite)
    monkeypatch.setattr(
        inference_route.ModelConfig,
        "from_identifier",
        lambda **_: config,
    )

    result = asyncio.run(
        inference_route.load_model(request, fastapi_request = None, current_subject = "t")
    )

    assert bridge_calls == [
        {
            "hf_repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
            "hf_variant": "Q4_K_M",
            "hf_token": "hf_test",
        }
    ]
    assert fake_termite.load_calls == [
        {
            "model_identifier": "unsloth/Llama-3.2-1B-Instruct-GGUF",
            "hf_token": "hf_test",
            "n_ctx": 8192,
        }
    ]
    assert result.model == "unsloth/Llama-3.2-1B-Instruct-GGUF"
    assert result.status == "loaded"


def test_openai_chat_route_streams_non_empty_termite_deltas(monkeypatch):
    inference_route = _load_route_module(
        "inference_route_module_for_termite_streaming_test",
        "routes/inference.py",
    )

    from models.inference import ChatCompletionRequest

    fake_termite = _FakeTermite(["Hel", "Hello", "Hello there"])

    monkeypatch.setattr(inference_route, "get_backend_kind", lambda: "termite-zig")
    monkeypatch.setattr(inference_route, "get_termite_backend", lambda: fake_termite)

    payload = ChatCompletionRequest(
        model = "current",
        messages = [{"role": "user", "content": "hello"}],
        stream = True,
        max_tokens = 16,
    )

    response = asyncio.run(
        inference_route.openai_chat_completions(payload, _FakeRequest())
    )
    chunks = asyncio.run(_collect_streaming_response(response))
    events = _sse_payloads(chunks)

    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    content_deltas = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if not e.get("_done") and e.get("choices")
    ]
    assert [d for d in content_deltas if d] == ["Hel", "lo", " there"]
    assert events[-1] == {"_done": True}


def test_openai_chat_route_non_stream_returns_full_termite_completion(monkeypatch):
    inference_route = _load_route_module(
        "inference_route_module_for_termite_nonstream_test",
        "routes/inference.py",
    )

    from models.inference import ChatCompletionRequest

    fake_termite = _FakeTermite(["Hi", "Hi there"])

    monkeypatch.setattr(inference_route, "get_backend_kind", lambda: "termite-zig")
    monkeypatch.setattr(inference_route, "get_termite_backend", lambda: fake_termite)

    payload = ChatCompletionRequest(
        model = "current",
        messages = [{"role": "user", "content": "hello"}],
        stream = False,
        max_tokens = 16,
    )

    body = asyncio.run(
        inference_route.openai_chat_completions(payload, _FakeRequest())
    )

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "Hi there"


def test_openai_chat_route_uses_termite_even_when_tools_are_enabled(monkeypatch):
    inference_route = _load_route_module(
        "inference_route_module_for_termite_tools_routing_test",
        "routes/inference.py",
    )

    from models.inference import ChatCompletionRequest

    fake_termite = _FakeTermite(["H", "Hi"])

    monkeypatch.setattr(inference_route, "get_backend_kind", lambda: "termite-zig")
    monkeypatch.setattr(inference_route, "get_termite_backend", lambda: fake_termite)
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: _ExplodingLlama(),
    )

    payload = ChatCompletionRequest(
        model = "current",
        messages = [{"role": "user", "content": "hello"}],
        stream = True,
        max_tokens = 16,
        enable_tools = True,
        enabled_tools = ["python"],
    )

    response = asyncio.run(
        inference_route.openai_chat_completions(payload, _FakeRequest())
    )
    chunks = asyncio.run(_collect_streaming_response(response))
    events = _sse_payloads(chunks)

    content_deltas = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if not e.get("_done") and e.get("choices")
    ]
    assert [d for d in content_deltas if d] == ["H", "i"]
