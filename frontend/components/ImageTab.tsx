"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, Loader2, RefreshCcw } from "lucide-react";
import DropZone from "./DropZone";
import DetectionBadge from "./DetectionBadge";
import DetectionTable from "./DetectionTable";
import { detectImage } from "@/lib/api";
import type { ImageDetectionResult } from "@/lib/types";

interface ImageTabProps {
  conf: number;
}

export default function ImageTab({ conf }: ImageTabProps) {
  const [result, setResult] = useState<ImageDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");

  const handleFile = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      setResult(null);
      setFileName(file.name);
      try {
        const res = await detectImage(file, conf);
        setResult(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Detection failed");
      } finally {
        setLoading(false);
      }
    },
    [conf]
  );

  const reset = () => {
    setResult(null);
    setError(null);
    setFileName("");
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Upload zone — always visible until result */}
      {!result && (
        <div className="relative">
          <DropZone accept="image" onFile={handleFile} disabled={loading} />
          <AnimatePresence>
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 flex flex-col items-center justify-center gap-3 rounded-2xl"
                style={{ background: "rgba(10,10,15,0.85)", backdropFilter: "blur(6px)" }}
              >
                <Loader2
                  size={36}
                  className="animate-spin"
                  style={{ color: "#FF3B30" }}
                />
                <p className="text-sm font-medium" style={{ color: "#888" }}>
                  Running YOLO26m…
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Error state */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
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
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="flex flex-col gap-5"
          >
            {/* Header row */}
            <div className="flex flex-wrap items-center gap-3">
              <DetectionBadge
                variant={result.boxes.length > 0 ? "accident" : "clear"}
                label={
                  result.boxes.length > 0
                    ? `🚨 ACCIDENT DETECTED — ${result.boxes.length} detection${result.boxes.length > 1 ? "s" : ""}`
                    : "✅ No Accident Detected"
                }
              />
              <span className="text-xs" style={{ color: "#555" }}>
                {result.elapsed_ms}ms · {result.image_size.width}×{result.image_size.height}px · conf {conf.toFixed(2)}
              </span>
              <button
                onClick={reset}
                className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors hover:bg-white/10"
                style={{ color: "#888", border: "1px solid rgba(255,255,255,0.08)" }}
              >
                <RefreshCcw size={12} />
                New Image
              </button>
            </div>

            {/* Side-by-side images */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Original */}
              <div className="glass-card overflow-hidden">
                <div
                  className="px-4 py-3 border-b text-sm font-medium"
                  style={{ borderColor: "rgba(255,255,255,0.07)", color: "#888" }}
                >
                  Original
                </div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/jpeg;base64,${result.original_image}`}
                  alt="Original"
                  className="w-full h-auto object-contain"
                  style={{ maxHeight: "400px" }}
                />
              </div>

              {/* Annotated */}
              <div
                className="glass-card overflow-hidden"
                style={{
                  boxShadow: result.boxes.length > 0
                    ? "0 0 30px rgba(255,59,48,0.12)"
                    : "0 0 30px rgba(52,199,89,0.08)",
                }}
              >
                <div
                  className="px-4 py-3 border-b text-sm font-medium"
                  style={{ borderColor: "rgba(255,255,255,0.07)", color: "#888" }}
                >
                  Annotated
                </div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image}`}
                  alt="Annotated"
                  className="w-full h-auto object-contain"
                  style={{ maxHeight: "400px" }}
                />
              </div>
            </div>

            {/* Download */}
            <a
              href={`data:image/jpeg;base64,${result.annotated_image}`}
              download={`accident_detected_${fileName}`}
              className="flex items-center justify-center gap-2 w-full py-3 rounded-xl font-semibold text-sm transition-all"
              style={{
                background: "linear-gradient(135deg, #FF3B30, #c0392b)",
                color: "white",
                boxShadow: "0 4px 20px rgba(255,59,48,0.35)",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 28px rgba(255,59,48,0.55)";
                (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.boxShadow = "0 4px 20px rgba(255,59,48,0.35)";
                (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
              }}
            >
              <Download size={16} />
              Download Annotated Image
            </a>

            {/* Detection table */}
            {result.boxes.length > 0 ? (
              <DetectionTable boxes={result.boxes} />
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-card p-5 text-sm text-center"
                style={{ color: "#555" }}
              >
                No detections above confidence threshold {conf.toFixed(2)}. Try lowering the slider in the sidebar.
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {!result && !loading && !error && (
        <p className="text-sm text-center" style={{ color: "#444" }}>
          👆 Upload an image to get started
        </p>
      )}
    </div>
  );
}
