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
import { getBackend, getBackendVersion, setBackend } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { BackendKind, BackendVersionInfo } from "../types/api";

/**
 * Unobtrusive inference-engine picker. Default is ``llama-cpp`` (the
 * existing GGUF flow); ``termite-zig`` routes chat through the Zig
 * server in ./termite-zig for dogfooding.
 *
 * When termite-zig is selected, the component also surfaces the
 * termite build's version + enabled backends (e.g. native / mlx / onnx)
 * so the user can tell at a glance which build is serving requests.
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
  const [versionInfo, setVersionInfo] = useState<BackendVersionInfo | null>(
    null,
  );

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

  // Fetch termite version whenever the picker flips to termite-zig.
  // Keeps the subline ("termite-zig v0.1.0 · native, mlx") accurate
  // without requiring a full page refresh.
  useEffect(() => {
    if (backendKind !== "termite-zig") {
      setVersionInfo(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const result = await getBackendVersion();
        if (!cancelled) {
          setVersionInfo(result.version);
        }
      } catch {
        if (!cancelled) {
          setVersionInfo(null);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [backendKind]);

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
            ? `Unloaded ${result.unloaded}. Pick a model to continue.`
            : "Pick a model to continue.",
        });
      } else {
        toast.info("Inference engine: llama.cpp", {
          description: result.unloaded
            ? `Unloaded ${result.unloaded}. Pick a model to continue.`
            : "Pick a model to continue.",
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

  const versionLine = formatVersionLine(backendKind, versionInfo);

  return (
    <div className="flex flex-col gap-1 px-2 py-2">
      <div className="flex items-center justify-between gap-3">
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
      {versionLine !== null && (
        <div
          className="text-[10px] text-muted-foreground/80 truncate"
          title={versionLine}
        >
          {versionLine}
        </div>
      )}
    </div>
  );
}

function formatVersionLine(
  kind: BackendKind,
  info: BackendVersionInfo | null,
): string | null {
  if (kind !== "termite-zig") return null;
  if (!info) return "termite-zig · version unavailable";
  const version = info.version ?? "unknown";
  const runtime = info.runtime ?? "termite-zig";
  const enabledBackends = info.backends
    ? Object.entries(info.backends)
        .filter(([, enabled]) => enabled)
        .map(([name]) => name)
    : [];
  const tail = enabledBackends.length
    ? ` · ${enabledBackends.join(", ")}`
    : "";
  return `${runtime} v${version}${tail}`;
}
