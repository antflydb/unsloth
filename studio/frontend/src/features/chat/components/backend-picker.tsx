// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { getBackend, setBackend } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { BackendKind } from "../types/api";

/**
 * Unobtrusive inference-engine picker. Default is ``llama-cpp`` (the
 * existing GGUF flow); ``termite-zig`` routes chat through the Zig
 * server in ./termite-zig for dogfooding.
 *
 * Disabled while a model is loading to avoid the user flipping the
 * backend mid-load.
 */
export function BackendPicker() {
  const backendKind = useChatRuntimeStore((s) => s.backendKind);
  const setBackendKind = useChatRuntimeStore((s) => s.setBackendKind);
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const clearCheckpoint = useChatRuntimeStore((s) => s.clearCheckpoint);
  const [pending, setPending] = useState(false);

  // Reconcile server-authoritative state on mount. Local storage is
  // a bootstrap hint; the server is authoritative (it's the thing
  // that actually has the subprocess). One-shot — no polling.
  const reconciledRef = useRef(false);
  useEffect(() => {
    if (reconciledRef.current) return;
    reconciledRef.current = true;
    let cancelled = false;
    (async () => {
      try {
        const result = await getBackend();
        if (!cancelled) {
          useChatRuntimeStore.getState().setBackendKind(result.backend);
        }
      } catch {
        // Non-fatal: if the call fails, stick with the localStorage value.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const onChange = async (next: string) => {
    if (next === backendKind || pending) return;
    if (next !== "llama-cpp" && next !== "termite-zig") return;
    const nextKind = next as BackendKind;

    setPending(true);
    try {
      const result = await setBackend({ backend: nextKind });
      setBackendKind(result.backend);
      // Model registries differ between backends; clear the current
      // checkpoint so the model picker repopulates from the new source.
      clearCheckpoint();
      if (nextKind === "termite-zig") {
        toast.info("Inference engine: termite-zig", {
          description: result.unloaded
            ? `Unloaded ${result.unloaded} from llama.cpp. Pick a termite model to continue.`
            : "Pick a model from the termite registry to continue.",
        });
      } else {
        toast.info("Inference engine: llama.cpp", {
          description: result.unloaded
            ? `Unloaded ${result.unloaded} from termite-zig. Pick a GGUF model to continue.`
            : "Pick a GGUF model to continue.",
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      toast.error("Could not switch inference engine", {
        description: message,
      });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex items-center justify-between gap-3 px-2 py-2">
      <div className="min-w-0">
        <div className="text-xs font-medium">Inference engine</div>
        <div className="text-[11px] text-muted-foreground">
          Which backend serves chat completions.
        </div>
      </div>
      <Select
        value={backendKind}
        onValueChange={onChange}
        disabled={modelLoading || pending}
      >
        <SelectTrigger
          size="sm"
          className="w-[9.5rem]"
          aria-label="Inference engine"
        >
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="llama-cpp">llama.cpp</SelectItem>
          <SelectItem value="termite-zig">termite-zig</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
