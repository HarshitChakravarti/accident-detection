"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Download, RefreshCcw, Loader2 } from "lucide-react";
import { Progress, ProgressTrack, ProgressIndicator } from "@/components/ui/progress";
import DropZone from "./DropZone";
import DetectionBadge from "./DetectionBadge";
import StatsGrid from "./StatsGrid";
import { detectVideoSSE, getVideoDownloadUrl } from "@/lib/api";
import type { VideoCompleteEvent, VideoProgress } from "@/lib/types";

interface VideoTabProps {
  conf: number;
  smoothing: number;
}

type Phase = "idle" | "uploading" | "processing" | "done" | "error";

interface ProgressState {
  pct: number;
  frame: number;
  totalFrames: number;
  rawDetections: number;
  confirmedIncidents: number;
}

export default function VideoTab({ conf, smoothing }: VideoTabProps) {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [result, setResult] = useState<VideoCompleteEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const handleFile = useCallback((f: File) => {
    // Revoke previous blob URL
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    const url = URL.createObjectURL(f);
    setFile(f);
    setFileUrl(url);
    setPhase("idle");
    setProgress(null);
    setResult(null);
    setError(null);
  }, [fileUrl]);

  const runDetection = async () => {
    if (!file) return;
    abortRef.current = new AbortController();
    setPhase("processing");
    setProgress(null);
    setResult(null);
    setError(null);

    try {
      const complete = await detectVideoSSE(
        file,
        conf,
        smoothing,
        (p: VideoProgress) => {
          setProgress({
            pct: p.progress,
            frame: p.frame,
            totalFrames: p.totalFrames,
            rawDetections: p.rawDetections,
            confirmedIncidents: p.confirmedIncidents,
          });
        },
        abortRef.current.signal
      );
      setResult(complete);
      setPhase("done");
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      setError(e instanceof Error ? e.message : "Processing failed");
      setPhase("error");
    }
  };

  const reset = () => {
    abortRef.current?.abort();
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFile(null);
    setFileUrl(null);
    setPhase("idle");
    setProgress(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="flex flex-col gap-6">

      {/* Drop zone */}
      {!file && (
        <DropZone accept="video" onFile={handleFile} />
      )}

      {/* File loaded state */}
      {file && phase !== "done" && (
        <div className="flex flex-col gap-4">
          {/* Source video preview */}
          <div className="glass-card overflow-hidden">
            <div
              className="px-4 py-3 border-b flex items-center justify-between"
              style={{ borderColor: "rgba(255,255,255,0.07)" }}
            >
              <p className="text-sm font-medium" style={{ color: "#888" }}>
                Source Video
              </p>
              <button
                onClick={reset}
                className="flex items-center gap-1 text-xs transition-colors hover:text-white/70"
                style={{ color: "#555" }}
              >
                <RefreshCcw size={12} />
                Change
              </button>
            </div>
            {fileUrl && (
              <video
                src={fileUrl}
                controls
                className="w-full"
                style={{ maxHeight: "320px", background: "#000" }}
              />
            )}
          </div>

          {/* Run button */}
          {phase === "idle" && (
            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={runDetection}
              className="flex items-center justify-center gap-2 w-full py-3.5 rounded-xl font-semibold text-sm"
              style={{
                background: "linear-gradient(135deg, #FF3B30, #c0392b)",
                color: "white",
                boxShadow: "0 4px 20px rgba(255,59,48,0.35)",
              }}
            >
              <Play size={16} fill="white" />
              Run Accident Detection
            </motion.button>
          )}

          {/* Progress */}
          <AnimatePresence>
            {phase === "processing" && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="glass-card p-5 flex flex-col gap-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Loader2 size={16} className="animate-spin" style={{ color: "#FF3B30" }} />
                    <span className="text-sm font-medium">Processing…</span>
                  </div>
                  <span
                    className="text-sm font-bold tabular-nums"
                    style={{ color: "#FF3B30" }}
                  >
                    {progress?.pct ?? 0}%
                  </span>
                </div>

                <Progress value={progress?.pct ?? 0}>
                  <ProgressTrack className="h-1.5">
                    <ProgressIndicator />
                  </ProgressTrack>
                </Progress>

                {progress && (
                  <p className="text-xs font-mono" style={{ color: "#666" }}>
                    Frame {progress.frame}/{progress.totalFrames} · Raw detections:{" "}
                    {progress.rawDetections} · Confirmed incidents:{" "}
                    <span style={{ color: progress.confirmedIncidents > 0 ? "#FF3B30" : "#666" }}>
                      {progress.confirmedIncidents}
                    </span>
                  </p>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-card p-4 text-sm"
            style={{ borderColor: "rgba(255,59,48,0.3)", color: "#FF3B30" }}
          >
            ❌ {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Results */}
      <AnimatePresence>
        {phase === "done" && result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="flex flex-col gap-5"
          >
            {/* Reset button */}
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm" style={{ color: "#888" }}>
                📊 Detection Summary
              </h3>
              <button
                onClick={reset}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors hover:bg-white/10"
                style={{ color: "#888", border: "1px solid rgba(255,255,255,0.08)" }}
              >
                <RefreshCcw size={12} />
                New Video
              </button>
            </div>

            {/* Stats grid */}
            <StatsGrid
              stats={[
                {
                  label: "Total Frames",
                  value: result.stats.totalFrames,
                },
                {
                  label: "Raw Detections",
                  value: result.stats.rawDetections,
                  hint: "Before temporal smoothing",
                },
                {
                  label: `Confirmed (N=${smoothing})`,
                  value: result.stats.confirmedIncidents,
                  hint: `${smoothing} consecutive frames`,
                  accent: true,
                },
                {
                  label: "Avg Confidence",
                  value: result.stats.avgConfidence,
                },
              ]}
            />

            {/* Caption */}
            <p className="text-xs" style={{ color: "#555" }}>
              Duration: {result.stats.durationSec} · FPS: {result.stats.fps} · Conf: {conf.toFixed(2)} · Smoothing N={smoothing}
            </p>

            {/* Badge */}
            <DetectionBadge
              variant={result.stats.confirmedIncidents > 0 ? "accident" : "clear"}
              label={
                result.stats.confirmedIncidents > 0
                  ? `🚨 ${result.stats.confirmedIncidents} confirmed incident${result.stats.confirmedIncidents > 1 ? "s" : ""} · ${result.stats.rawDetections} raw detections`
                  : "✅ No Accidents Detected"
              }
            />

            {/* Output video player */}
            {result.stats.videoAvailable && (
              <div className="glass-card overflow-hidden">
                <div
                  className="px-4 py-3 border-b text-sm font-medium"
                  style={{ borderColor: "rgba(255,255,255,0.07)", color: "#888" }}
                >
                  🎬 Annotated Output
                </div>
                <video
                  src={getVideoDownloadUrl(result.jobId)}
                  controls
                  className="w-full"
                  style={{ maxHeight: "480px", background: "#000" }}
                />
              </div>
            )}

            {/* Download button */}
            {result.stats.videoAvailable && (
              <a
                href={getVideoDownloadUrl(result.jobId)}
                download={`accident_detected_${file?.name ?? "video.mp4"}`}
                className="flex items-center justify-center gap-2 w-full py-3 rounded-xl font-semibold text-sm transition-all"
                style={{
                  background: "linear-gradient(135deg, #FF3B30, #c0392b)",
                  color: "white",
                  boxShadow: "0 4px 20px rgba(255,59,48,0.35)",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 28px rgba(255,59,48,0.55)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 20px rgba(255,59,48,0.35)";
                }}
              >
                <Download size={16} />
                Download Annotated Video
              </a>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!file && (
        <p className="text-sm text-center" style={{ color: "#444" }}>
          👆 Upload a video to get started
        </p>
      )}
    </div>
  );
}
