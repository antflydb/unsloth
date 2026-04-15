# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference submodule - Inference backend for model loading and generation

The default get_inference_backend() returns an InferenceOrchestrator that
delegates to a subprocess. The original InferenceBackend runs inside
the subprocess and can be imported directly from .inference when needed.
"""

from .orchestrator import InferenceOrchestrator, get_inference_backend
from .llama_cpp import LlamaCppBackend
from .backend_state import (
    BackendKind,
    get_backend_kind,
    get_termite_backend,
    set_backend_kind,
)

# Expose InferenceOrchestrator as InferenceBackend for backward compat
InferenceBackend = InferenceOrchestrator

__all__ = [
    "InferenceBackend",
    "InferenceOrchestrator",
    "get_inference_backend",
    "LlamaCppBackend",
    "BackendKind",
    "get_backend_kind",
    "set_backend_kind",
    "get_termite_backend",
]
