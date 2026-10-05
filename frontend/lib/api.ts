// lib/api.ts — typed API client for the FastAPI backend

import type {
  HealthResponse,
  ImageDetectionResult,
  VideoCompleteEvent,
  VideoProgress,
  VideoSSEEvent,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Health ─────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Backend health check failed");
  return res.json();
}

// ─── Image Detection ────────────────────────────────────────────────────────

export async function detectImage(
  file: File,
  conf: number
): Promise<ImageDetectionResult> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("conf", conf.toString());

  const res = await fetch(`${BASE}/detect/image`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Image detection failed");
  }

  return res.json();
}

// ─── Video Detection (SSE) ──────────────────────────────────────────────────

/**
 * Stream video detection events from the backend.
 * Calls onProgress for each progress event, resolves with the complete event.
 */
export function detectVideoSSE(
  file: File,
  conf: number,
  smoothing: number,
  onProgress: (p: VideoProgress) => void,
  signal?: AbortSignal
): Promise<VideoCompleteEvent> {
  return new Promise(async (resolve, reject) => {
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("conf", conf.toString());
      fd.append("smoothing", smoothing.toString());

      const res = await fetch(`${BASE}/detect/video`, {
        method: "POST",
        body: fd,
        signal,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        reject(new Error((err as { detail?: string }).detail ?? "Video detection failed"));
        return;
      }

      if (!res.body) {
        reject(new Error("No response body for SSE stream"));
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event: VideoSSEEvent = JSON.parse(line.slice(6));
              if (event.type === "progress") {
                onProgress(event as VideoProgress);
              } else if (event.type === "complete") {
                resolve(event as VideoCompleteEvent);
                return;
              } else if (event.type === "error") {
                reject(new Error((event as { type: string; message: string }).message));
                return;
              }
            } catch {
              // skip malformed SSE lines
            }
          }
        }
      }
    } catch (err) {
      reject(err);
    }
  });
}

// ─── Video Download URL ─────────────────────────────────────────────────────

export function getVideoDownloadUrl(jobId: string): string {
  return `${BASE}/detect/video/download/${jobId}`;
}
