# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
termite-zig chat completions helper.

Kept as a small standalone module so ``routes/inference.py`` only
needs a 3-line early-return at the top of ``openai_chat_completions``
to dispatch to termite — the rest of the existing llama.cpp / Unsloth
code stays byte-identical for upstream merges.

The helper consumes ``TermiteZigBackend.generate_chat_completion``
(cumulative yields) and re-emits OpenAI-style SSE deltas so frontends
see the same wire format as the llama.cpp path.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Any, AsyncGenerator

from loggers import get_logger

logger = get_logger(__name__)

_STREAM_SENTINEL = object()


async def stream_termite_chat(
    *,
    payload: Any,
    request: Any,
    termite_backend: Any,
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chat completion chunks from a termite backend.

    Arguments:
        payload:         ``ChatCompletionRequest``-shaped object (uses
                         ``.messages``, ``.temperature`` etc.).
        request:         FastAPI ``Request`` — used for disconnect detection.
        termite_backend: ``TermiteZigBackend`` instance with an already
                         selected model.

    Raises ``RuntimeError`` if the backend has no selected model — this
    surfaces as a 500 in the route, prompting the user to pick a model.
    """
    if not termite_backend.is_loaded or not termite_backend.model_identifier:
        raise RuntimeError(
            "termite-zig: no model selected. Pick a model in the Studio UI first."
        )

    model_name = termite_backend.model_identifier
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    cancel_event = threading.Event()

    # Role preamble — matches the llama.cpp SSE stream so the frontend
    # adapter doesn't need to special-case termite.
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
        ],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    # Coerce messages into plain dicts — the termite backend accepts
    # whatever dict shape the caller hands over.
    messages = [
        m if isinstance(m, dict) else m.model_dump(exclude_none = True)
        for m in payload.messages
    ]

    gen = termite_backend.generate_chat_completion(
        messages = messages,
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_tokens = payload.max_tokens,
        repetition_penalty = payload.repetition_penalty,
        presence_penalty = payload.presence_penalty,
        cancel_event = cancel_event,
        enable_thinking = payload.enable_thinking,
    )

    prev_text = ""
    try:
        while True:
            # Disconnect check: if the client went away, stop driving
            # the backend so it doesn't waste tokens into the void.
            if await request.is_disconnected():
                cancel_event.set()
                break

            cumulative = await asyncio.to_thread(next, gen, _STREAM_SENTINEL)
            if cumulative is _STREAM_SENTINEL:
                break
            if not isinstance(cumulative, str):
                # v1: termite only yields strings. If future versions
                # yield metadata dicts (usage, timings), skip them for now.
                continue

            new_text = cumulative[len(prev_text):]
            prev_text = cumulative
            if not new_text:
                continue

            delta_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(delta_chunk)}\n\n"

    finally:
        # Close the generator so the HTTP stream is cleanly released
        # regardless of how we exit (completion, disconnect, exception).
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            pass

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
