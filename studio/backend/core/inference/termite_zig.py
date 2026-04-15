# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
termite-zig inference backend for Studio chat.

Wraps the Zig reimplementation of Antfly's termite server (see
``termite-zig/`` in this repo). Exposes the same public surface as
``LlamaCppBackend`` so ``routes/inference.py`` can dispatch on
``backend_state.get_backend_kind()`` with tiny, easy-to-merge
conditional guards.

v1 scope (per the picker plan): text streaming + model list. No tool
translation, no image inputs, no load-progress parity.

Launch model:
  * If ``TERMITE_ZIG_URL`` is set, connect to that externally-managed
    server and skip spawning — this is how we attach under a debugger
    or point at a manually-built termite.
  * Otherwise, auto-spawn ``<binary> run --port <free-port>`` via
    ``subprocess.Popen`` and poll ``/readyz`` until the server is up.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator, Optional

import httpx

from loggers import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Repo root — 4 parents up from this file (studio/backend/core/inference/).
_REPO_ROOT: str = str(Path(__file__).resolve().parents[4])

# Polling config for ``/readyz``. Tests override these to keep the
# retry loop cheap; real defaults give termite enough time to load its
# ONNX runtime and warm the registry on a fresh machine.
_READY_MAX_WAIT_SECONDS: float = 60.0
_READY_POLL_INTERVAL_SECONDS: float = 0.5

# Env vars.
_ENV_BINARY_OVERRIDE = "TERMITE_BIN"
_ENV_URL_OVERRIDE = "TERMITE_ZIG_URL"


def _find_free_port() -> int:
    """Ask the kernel for a free TCP port (same trick as llama_cpp)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Backend class ─────────────────────────────────────────────────────


class TermiteZigBackend:
    """Runs and proxies to a termite-zig server subprocess.

    Mirrors ``LlamaCppBackend``'s public surface so the dispatcher in
    ``routes/inference.py`` can switch between backends with minimal
    branching.
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._external_url: Optional[str] = None
        self._model_identifier: Optional[str] = None
        self._healthy: bool = False
        self._lock = threading.Lock()
        # Ensure spawned subprocess doesn't outlive the Studio server.
        atexit.register(self._cleanup)

    # ── Properties (mirror LlamaCppBackend) ─────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True once a model identifier has been selected AND the server is ready."""
        return self._healthy and self._model_identifier is not None

    @property
    def is_active(self) -> bool:
        """True if the termite server is running (spawned or external)."""
        return self._healthy

    @property
    def base_url(self) -> str:
        """Base URL for HTTP calls to termite, without the /ml/v1 prefix."""
        if self._external_url:
            return self._external_url
        if self._port is None:
            return ""
        return f"http://127.0.0.1:{self._port}"

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    # ── Binary discovery ─────────────────────────────────────────────

    @staticmethod
    def _find_termite_binary() -> Optional[str]:
        """Locate the termite binary.

        Search order:
          1. ``TERMITE_BIN`` env var (explicit path).
          2. ``<repo>/termite-zig/zig-out/bin/termite`` (dev build inside
             this monorepo — the common case when developing locally).
          3. ``~/.unsloth/termite/bin/termite`` (installed copy).
          4. ``termite`` on PATH (system install).
        """
        binary_name = "termite.exe" if sys.platform == "win32" else "termite"

        env_path = os.environ.get(_ENV_BINARY_OVERRIDE)
        if env_path and Path(env_path).is_file():
            return env_path

        repo_build = Path(_REPO_ROOT) / "termite-zig" / "zig-out" / "bin" / binary_name
        if repo_build.is_file():
            return str(repo_build)

        # ``./termite`` at the unsloth repo root — common when the user
        # builds termite-zig and copies / symlinks the binary up here.
        repo_root_binary = Path(_REPO_ROOT) / binary_name
        if repo_root_binary.is_file():
            return str(repo_root_binary)

        home_build = Path.home() / ".unsloth" / "termite" / "bin" / binary_name
        if home_build.is_file():
            return str(home_build)

        system_path = shutil.which(binary_name)
        if system_path:
            return system_path

        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def _ensure_running(self) -> None:
        """Ensure the termite server is reachable.

        Respects ``TERMITE_ZIG_URL`` for bring-your-own-termite mode
        (debugger attach, custom build). Otherwise spawns a subprocess
        on a free port and polls ``/readyz``.
        """
        with self._lock:
            if self._healthy:
                return

            url_override = os.environ.get(_ENV_URL_OVERRIDE)
            if url_override:
                self._external_url = url_override.rstrip("/")
                self._wait_until_ready()
                return

            binary = self._find_termite_binary()
            if binary is None:
                raise RuntimeError(
                    "termite binary not found. Build it with `zig build` "
                    f"under ./termite-zig, install to ~/.unsloth/termite/bin, "
                    f"or set {_ENV_BINARY_OVERRIDE}=<path>."
                )

            port = _find_free_port()
            self._port = port
            logger.info("Spawning termite-zig subprocess: %s run --port %d", binary, port)
            self._process = subprocess.Popen(
                [binary, "run", "--port", str(port)],
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
            )
            try:
                self._wait_until_ready()
            except Exception:
                # Reap the failed subprocess so a follow-up retry has a
                # clean slate — otherwise we'd leak a hung termite.
                self._terminate_process()
                raise

    def _wait_until_ready(self) -> None:
        """Poll ``/ml/v1/healthz`` until 200, or time out.

        All operational endpoints live under the ``/ml/v1/`` prefix
        (unlike the original Go termite, which exposed them at root).
        ``/healthz`` tells us the HTTP server is alive; ``/readyz``
        adds "has discovered at least one model" which is too strict
        for us — we may want to fire up termite with an empty model
        directory and let the UI guide the user to ``termite pull``.
        """
        deadline = time.monotonic() + _READY_MAX_WAIT_SECONDS
        url = f"{self.base_url}/ml/v1/healthz"
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout = 2.0)
                if resp.status_code == 200:
                    self._healthy = True
                    return
                last_err = RuntimeError(f"{url} returned {resp.status_code}")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(_READY_POLL_INTERVAL_SECONDS)

        raise RuntimeError(
            f"termite-zig at {self.base_url} not ready after "
            f"{_READY_MAX_WAIT_SECONDS}s: {last_err}"
        )

    def version(self) -> Optional[dict]:
        """Return termite's ``/ml/v1/version`` payload, or None on failure.

        Used by the frontend to surface build info next to the picker
        (e.g. ``termite-zig v0.1.0 \u2022 backends: native, mlx``). Non-fatal
        if the call fails; the picker keeps working.
        """
        if not self._healthy:
            return None
        try:
            resp = httpx.get(f"{self.base_url}/ml/v1/version", timeout = 2.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("termite-zig version probe failed: %s", exc)
        return None

    def _terminate_process(self) -> None:
        proc = self._process
        self._process = None
        self._port = None
        self._healthy = False
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout = 5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout = 5.0)
        except Exception:  # noqa: BLE001
            pass

    def _cleanup(self) -> None:
        self._terminate_process()

    # ── Model lifecycle ──────────────────────────────────────────────

    def load_model(
        self,
        *,
        model_identifier: str,
        hf_token: Optional[str] = None,
        n_ctx: int = 0,
        **_unused: Any,
    ) -> bool:
        """Record the identifier and let termite lazy-load on first chat.

        termite-zig discovers models from its models dir on each
        registry read, and loads weights on the first chat request.
        The only eager work here is (a) ensure termite is running and
        (b) cache the identifier so ``is_loaded`` / ``model_identifier``
        behave like the llama.cpp backend.

        No registry validation: the caller (routes/inference.py) is
        responsible for bridging the HF cache into termite's layout
        before calling this, and a mismatched id will surface as a
        useful error on the first chat request.
        """
        self._ensure_running()
        self._model_identifier = model_identifier
        logger.info(
            "termite-zig: selected model %s (lazy-load on first request)",
            model_identifier,
        )
        return True

    def unload_model(self) -> bool:
        """Clear the active model identifier.

        termite-zig keeps models warm in its own LRU — there is no
        explicit unload endpoint in v1. Clearing the cached identifier
        here is enough for Studio's dispatcher: the next load_model
        call will validate a new one.
        """
        self._model_identifier = None
        return True

    # ── Model listing ────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        """Return termite's model registry as OpenAI-style dicts.

        Returns the raw ``data`` entries from ``GET /ml/v1/models`` so
        the caller can apply Studio-specific filtering (vision /
        embedding / reranker) without this class knowing about them.
        """
        if not self._healthy:
            self._ensure_running()
        url = f"{self.base_url}/ml/v1/models"
        resp = httpx.get(url, timeout = 10.0)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        return list(data)

    # ── Chat completions ─────────────────────────────────────────────

    def generate_chat_completion(
        self,
        messages: list[dict],
        *,
        image_b64: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        **_ignored: Any,
    ) -> Generator[str, None, None]:
        """Stream a chat response from termite, yielding cumulative text.

        Matches ``LlamaCppBackend.generate_chat_completion``'s contract:
        each yield is the full text so far. That lets ``chat-adapter.ts``
        treat both backends identically.
        """
        if self._model_identifier is None:
            raise RuntimeError("termite-zig: no model selected — call load_model first")
        self._ensure_running()

        body: dict[str, Any] = {
            "model": self._model_identifier,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            body["top_k"] = top_k
        if min_p is not None:
            body["min_p"] = min_p
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if repetition_penalty is not None:
            body["repetition_penalty"] = repetition_penalty
        if presence_penalty is not None:
            body["presence_penalty"] = presence_penalty
        if stop:
            body["stop"] = stop

        accumulated: str = ""
        url = f"{self.base_url}/ml/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        with httpx.Client(timeout = None) as client:
            with client.stream("POST", url, json = body, headers = headers) as resp:
                if resp.status_code >= 400:
                    # Drain the response so we can surface termite's own
                    # error body (e.g. ``NoTokenizerFound``) rather than
                    # raising a bare HTTPStatusError with no context.
                    body_bytes = b""
                    try:
                        for chunk in resp.iter_bytes():
                            body_bytes += chunk
                            if len(body_bytes) > 4096:
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    detail = body_bytes.decode("utf-8", errors = "replace").strip()
                    raise RuntimeError(
                        f"termite-zig /ml/v1/chat/completions returned "
                        f"{resp.status_code}: {detail or '(empty body)'}"
                    )
                for raw_line in resp.iter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("termite-zig: skipping malformed SSE chunk: %s", data[:200])
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if not piece:
                        continue
                    accumulated += piece
                    yield accumulated

    # ── Load progress (stub) ─────────────────────────────────────────

    def load_progress(self) -> Optional[dict]:
        """No equivalent for termite-zig in v1.

        The UI tolerates ``None`` by falling back to its generic
        spinner (see the load-progress fallback path in
        ``use-chat-model-runtime.ts``). When termite grows a progress
        endpoint of its own, map it to the same
        ``{phase, bytes_loaded, bytes_total, fraction}`` shape here.
        """
        return None
