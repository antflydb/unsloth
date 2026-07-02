# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Antfly inference backend for Studio chat.

Wraps the Zig Antfly inference server (see ``antfly/zig`` in this
repo). Exposes the same public surface as ``LlamaCppBackend`` so
``routes/inference.py`` can dispatch on ``backend_state.get_backend_kind()``
with tiny, easy-to-merge conditional guards.

v1 scope (per the picker plan): text streaming + model list. No tool
translation, no image inputs, no load-progress parity.

Launch model:
  * If ``ANTFLY_INFERENCE_URL`` (or legacy ``TERMITE_ZIG_URL``) is set,
    connect to that externally-managed server and skip spawning.
  * Otherwise, auto-spawn ``<binary> inference run --port <free-port>`` via
    ``subprocess.Popen`` and poll root ``/healthz`` until the server is up.
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

# Polling config for root ``/healthz``. Tests override these to keep the
# retry loop cheap; real defaults give Antfly enough time to load its
# runtime and warm the registry on a fresh machine.
_READY_MAX_WAIT_SECONDS: float = 60.0
_READY_POLL_INTERVAL_SECONDS: float = 0.5

# Env vars.
_ENV_BINARY_OVERRIDE = "ANTFLY_BIN"
_ENV_BINARY_OVERRIDE_LEGACY = "TERMITE_BIN"
_ENV_URL_OVERRIDE = "ANTFLY_INFERENCE_URL"
_ENV_URL_OVERRIDE_LEGACY = "TERMITE_ZIG_URL"
_ENV_MODELS_DIR = "ANTFLY_MODELS_DIR"
_ENV_MODELS_DIR_LEGACY = "TERMITE_MODELS_DIR"
_API_PREFIX = "/ai/v1"


def _find_free_port() -> int:
    """Ask the kernel for a free TCP port (same trick as llama_cpp)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Backend class ─────────────────────────────────────────────────────


class TermiteZigBackend:
    """Runs and proxies to an Antfly inference server subprocess.

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
        """True if the Antfly inference server is running (spawned or external)."""
        return self._healthy

    @property
    def base_url(self) -> str:
        """Base URL for HTTP calls to Antfly inference, without an API prefix."""
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
        """Locate the Antfly inference binary.

        Search order:
          1. ``ANTFLY_BIN`` env var (explicit path), then legacy ``TERMITE_BIN``.
          2. ``<repo>/antfly/zig/zig-out/bin/antfly`` (current dev build).
          3. ``~/.unsloth/antfly/bin/antfly`` (installed copy).
          4. ``antfly`` on PATH.

        ``TERMITE_BIN`` remains accepted only as an explicit override so
        old local environments can point it at the new Antfly binary.
        """
        antfly_name = "antfly.exe" if sys.platform == "win32" else "antfly"

        for env_var in (_ENV_BINARY_OVERRIDE, _ENV_BINARY_OVERRIDE_LEGACY):
            env_path = os.environ.get(env_var)
            if env_path and Path(env_path).is_file():
                return env_path

        candidates = [
            Path(_REPO_ROOT) / "antfly" / "zig" / "zig-out" / "bin" / antfly_name,
            Path.home() / ".unsloth" / "antfly" / "bin" / antfly_name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        system_path = shutil.which(antfly_name)
        if system_path:
            return system_path

        return None

    # ── Lifecycle ────────────────────────────────────────────────────

    def _ensure_running(self) -> None:
        """Ensure the Antfly inference server is reachable.

        Respects ``ANTFLY_INFERENCE_URL`` (and legacy ``TERMITE_ZIG_URL``)
        for bring-your-own-server mode. Otherwise spawns a subprocess
        on a free port and polls root ``/healthz``.
        """
        with self._lock:
            self._bridge_hf_cache_models()
            if self._healthy:
                if self._probe_health():
                    return
                logger.warning(
                    "Antfly inference marked healthy but %s is unreachable; resetting backend state",
                    self.base_url or "(no url)",
                )
                self._healthy = False
                if self._process is not None:
                    self._terminate_process()

            url_override = os.environ.get(_ENV_URL_OVERRIDE) or os.environ.get(_ENV_URL_OVERRIDE_LEGACY)
            if url_override:
                self._external_url = url_override.rstrip("/")
                self._wait_until_ready()
                return

            binary = self._find_termite_binary()
            if binary is None:
                raise RuntimeError(
                    "Antfly inference binary not found. Build it with `zig build install` "
                    f"under ./antfly/zig, install to ~/.unsloth/antfly/bin, "
                    f"or set {_ENV_BINARY_OVERRIDE}=<path>."
                )

            port = _find_free_port()
            self._port = port
            logger.info("Spawning Antfly inference subprocess: %s inference run --port %d", binary, port)
            cmd = [binary, "inference", "run", "--port", str(port)]
            models_dir = os.environ.get(_ENV_MODELS_DIR) or os.environ.get(_ENV_MODELS_DIR_LEGACY)
            if not models_dir:
                from core.inference.termite_bridge import termite_models_dir

                models_dir = str(termite_models_dir())
            cmd.extend(["--models-dir", models_dir])
            self._process = subprocess.Popen(
                cmd,
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

    def _bridge_hf_cache_models(self) -> None:
        """Best-effort local HF-cache → Antfly model-layout bridge."""

        try:
            from core.inference.termite_bridge import bridge_hf_cache_to_termite

            bridge_hf_cache_to_termite()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Antfly inference cache bridge skipped: %s", exc)

    @staticmethod
    def _local_generator_model_entries() -> list[dict]:
        """Return synthetic OpenAI-style entries from Antfly's models dir.

        Antfly's native ``/ai/v1/models`` endpoint may do expensive
        loadability checks over large local GGUF directories. Studio already
        knows what it made visible to Antfly via the symlink bridge, so expose
        those generator ids immediately and merge the server's response when
        available.
        """

        try:
            from core.inference.termite_bridge import termite_models_dir

            root = termite_models_dir() / "generators"
            if not root.is_dir():
                return []
            entries: list[dict] = []
            for owner_dir in root.iterdir():
                if not owner_dir.is_dir() or owner_dir.name.startswith("."):
                    continue
                for model_dir in owner_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name.startswith("."):
                        continue
                    try:
                        has_model_file = any(
                            f.is_file() and f.name.endswith(".gguf")
                            for f in model_dir.iterdir()
                        )
                    except OSError:
                        continue
                    if not has_model_file:
                        continue
                    entries.append(
                        {
                            "id": f"{owner_dir.name}/{model_dir.name}",
                            "object": "model",
                            "created": 0,
                            "owned_by": "antfly",
                        }
                    )
            return sorted(entries, key = lambda e: e["id"].lower())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Antfly local model scan skipped: %s", exc)
            return []

    def _wait_until_ready(self) -> None:
        """Poll root ``/healthz`` until 200, or time out.

        Antfly inference keeps operational endpoints at the server root;
        public inference routes live under ``/ai/v1``. ``/readyz`` adds
        "has discovered at least one model" which is too strict for us —
        Studio may start with an empty models directory and guide the user
        to select/download a model.
        """
        deadline = time.monotonic() + _READY_MAX_WAIT_SECONDS
        url = f"{self.base_url}/healthz"
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
            f"Antfly inference at {self.base_url} not ready after "
            f"{_READY_MAX_WAIT_SECONDS}s: {last_err}"
        )

    def _probe_health(self) -> bool:
        """Best-effort liveness check for the current Antfly inference endpoint."""
        if self._process is not None and self._process.poll() is not None:
            return False
        if not self.base_url:
            return False
        try:
            resp = httpx.get(f"{self.base_url}/healthz", timeout = 2.0)
        except Exception:  # noqa: BLE001
            return False
        return resp.status_code == 200

    def version(self) -> Optional[dict]:
        """Return Antfly inference version payload, or None on failure.

        Used by the frontend to surface build info next to the picker
        (e.g. ``antfly-inference dev • backends: native, onnx``). Non-fatal
        if the call fails; the picker keeps working.
        """
        if not self._healthy:
            return None
        try:
            resp = httpx.get(f"{self.base_url}{_API_PREFIX}/version", timeout = 2.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Antfly inference version probe failed: %s", exc)
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
        """Record the identifier and let Antfly inference lazy-load on first chat.

        Antfly inference discovers models from its models dir on each
        registry read, and loads weights on the first chat request.
        The only eager work here is (a) ensure Antfly inference is running and
        (b) cache the identifier so ``is_loaded`` / ``model_identifier``
        behave like the llama.cpp backend.

        No registry validation: the caller (routes/inference.py) is
        responsible for bridging the HF cache into Antfly inference's layout
        before calling this, and a mismatched id will surface as a
        useful error on the first chat request.
        """
        self._ensure_running()
        self._model_identifier = model_identifier
        logger.info(
            "Antfly inference: selected model %s (lazy-load on first request)",
            model_identifier,
        )
        return True

    def unload_model(self) -> bool:
        """Clear the active model identifier.

        Antfly inference keeps models warm in its own LRU — there is no
        explicit unload endpoint in v1. Clearing the cached identifier
        here is enough for Studio's dispatcher: the next load_model
        call will validate a new one.
        """
        self._model_identifier = None
        return True

    # ── Model listing ────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        """Return Antfly inference's model registry as OpenAI-style dicts.

        Returns the raw ``data`` entries from ``GET /ai/v1/models`` so
        the caller can apply Studio-specific filtering (vision /
        embedding / reranker) without this class knowing about them.
        """
        if not self._healthy:
            self._ensure_running()
        else:
            self._bridge_hf_cache_models()
        data: list[dict] = []
        url = f"{self.base_url}{_API_PREFIX}/models"
        try:
            resp = httpx.get(url, timeout = 5.0)
            resp.raise_for_status()
            payload = resp.json()
            data = list(payload.get("data", []))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Antfly inference model registry request failed or timed out; "
                "falling back to local models-dir scan: %s",
                exc,
            )

        seen = {
            entry.get("id")
            for entry in data
            if isinstance(entry, dict) and entry.get("id")
        }
        for entry in self._local_generator_model_entries():
            if entry["id"] in seen:
                continue
            data.append(entry)
            seen.add(entry["id"])
        return data

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
        """Stream a chat response from Antfly inference, yielding cumulative text.

        Matches ``LlamaCppBackend.generate_chat_completion``'s contract:
        each yield is the full text so far. That lets ``chat-adapter.ts``
        treat both backends identically.
        """
        if self._model_identifier is None:
            raise RuntimeError("Antfly inference: no model selected — call load_model first")
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
        url = f"{self.base_url}{_API_PREFIX}/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        for attempt in range(2):
            if attempt:
                self._healthy = False
                if self._process is not None:
                    self._terminate_process()
                self._ensure_running()
                url = f"{self.base_url}{_API_PREFIX}/chat/completions"

            try:
                with httpx.Client(timeout = None) as client:
                    with client.stream("POST", url, json = body, headers = headers) as resp:
                        if resp.status_code >= 400:
                            # Drain the response so we can surface Antfly inference's own
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
                                f"Antfly inference {_API_PREFIX}/chat/completions returned "
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
                                logger.debug("Antfly inference: skipping malformed SSE chunk: %s", data[:200])
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
                        return
            except httpx.ConnectError:
                if attempt:
                    raise
                logger.warning(
                    "Antfly inference connection to %s refused during chat; retrying once",
                    url,
                )

    # ── Load progress (stub) ─────────────────────────────────────────

    def load_progress(self) -> Optional[dict]:
        """No equivalent for Antfly inference in v1.

        The UI tolerates ``None`` by falling back to its generic
        spinner (see the load-progress fallback path in
        ``use-chat-model-runtime.ts``). When Antfly inference grows a progress
        endpoint of its own, map it to the same
        ``{phase, bytes_loaded, bytes_total, fraction}`` shape here.
        """
        return None
