"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Slider } from "@/components/ui/slider";
import { CheckCircle, XCircle, ChevronDown, ChevronUp } from "lucide-react";
import { fetchHealth } from "@/lib/api";
import type { ModelStats } from "@/lib/types";

const MODEL_STATS_FALLBACK: ModelStats = {
  Architecture: "YOLO26m",
  "Epochs Trained": 100,
  "Training Images": "8,759",
  "mAP@0.50": "0.179",
  "mAP@0.50:95": "0.092",
  Precision: "0.707",
  Recall: "0.426",
  Parameters: "20.3 M",
};

interface SidebarProps {
  conf: number;
  setConf: (v: number) => void;
  smoothing: number;
  setSmoothing: (v: number) => void;
}

export default function Sidebar({ conf, setConf, smoothing, setSmoothing }: SidebarProps) {
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);
  const [modelStats, setModelStats] = useState<ModelStats>(MODEL_STATS_FALLBACK);
  const [modelInfoOpen, setModelInfoOpen] = useState(true);

  useEffect(() => {
    fetchHealth()
      .then((h) => {
        setModelLoaded(h.model_loaded);
        if (h.model_stats) setModelStats(h.model_stats);
      })
      .catch(() => setModelLoaded(false));
  }, []);

  return (
    <motion.aside
      initial={{ x: -40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-72 flex-shrink-0 h-screen sticky top-0 flex flex-col gap-4 overflow-y-auto p-5"
      style={{ background: "rgba(10,10,15,0.95)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 mb-2">
        <div
          className="text-2xl select-none"
          style={{ filter: "drop-shadow(0 0 12px rgba(255,59,48,0.7))" }}
        >
          🚨
        </div>
        <div>
          <h1 className="text-lg font-bold leading-tight" style={{ color: "#FF3B30" }}>
            Accident Detection
          </h1>
          <p className="text-xs" style={{ color: "#888" }}>
            Powered by YOLO26m
          </p>
        </div>
      </div>

      {/* Divider */}
      <div className="h-px" style={{ background: "rgba(255,255,255,0.07)" }} />

      {/* Settings */}
      <section className="glass-card p-4 flex flex-col gap-5">
        <h2 className="text-sm font-semibold tracking-wide uppercase" style={{ color: "#888" }}>
          ⚙️ Settings
        </h2>

        {/* Confidence Threshold */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium text-white/80">Confidence Threshold</label>
            <span
              className="text-sm font-bold tabular-nums px-2 py-0.5 rounded-md"
              style={{ background: "rgba(255,59,48,0.12)", color: "#FF3B30" }}
            >
              {conf.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0.30}
            max={0.90}
            step={0.05}
            value={[conf]}
            onValueChange={(v: number | readonly number[]) => {
              const arr = Array.isArray(v) ? v : [v];
              setConf((arr as number[])[0]);
            }}
          />
          <p className="text-xs" style={{ color: "#666" }}>
            0.60 recommended · Higher = fewer but more certain detections
          </p>
        </div>

        {/* Temporal Smoothing */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center">
            <label className="text-sm font-medium text-white/80">Temporal Smoothing</label>
            <span
              className="text-sm font-bold tabular-nums px-2 py-0.5 rounded-md"
              style={{
                background: smoothing === 1 ? "rgba(255,255,255,0.06)" : "rgba(255,59,48,0.12)",
                color: smoothing === 1 ? "#888" : "#FF3B30",
              }}
            >
              N = {smoothing}
            </span>
          </div>
          <Slider
            min={1}
            max={7}
            step={1}
            value={[smoothing]}
            onValueChange={(v: number | readonly number[]) => {
              const arr = Array.isArray(v) ? v : [v];
              setSmoothing((arr as number[])[0]);
            }}
          />
          <p className="text-xs" style={{ color: "#666" }}>
            {smoothing === 1
              ? "ℹ️ Smoothing OFF — every single-frame detection counts"
              : `✅ Need ${smoothing} consecutive frames to confirm incident`}
          </p>
        </div>
      </section>

      {/* Model Info */}
      <section className="glass-card overflow-hidden">
        <button
          onClick={() => setModelInfoOpen((o) => !o)}
          className="w-full flex items-center justify-between p-4 text-sm font-semibold tracking-wide uppercase cursor-pointer hover:bg-white/5 transition-colors"
          style={{ color: "#888" }}
        >
          <span>🤖 Model Info</span>
          {modelInfoOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        <AnimatePresence initial={false}>
          {modelInfoOpen && (
            <motion.div
              key="model-info"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-4 flex flex-col gap-1.5">
                {Object.entries(modelStats).map(([k, v]) => (
                  <div key={k} className="flex justify-between items-center text-sm">
                    <span style={{ color: "#888" }}>{k}</span>
                    <span className="font-mono font-semibold text-white/90">{v}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* Weights status */}
      <section className="glass-card p-4">
        <h2 className="text-sm font-semibold tracking-wide uppercase mb-3" style={{ color: "#888" }}>
          📂 Weights
        </h2>
        {modelLoaded === null && (
          <p className="text-sm" style={{ color: "#888" }}>Checking…</p>
        )}
        {modelLoaded === true && (
          <div className="flex items-center gap-2 text-sm" style={{ color: "#34C759" }}>
            <CheckCircle size={16} />
            <span className="font-medium">best.pt loaded</span>
          </div>
        )}
        {modelLoaded === false && (
          <div className="flex items-center gap-2 text-sm" style={{ color: "#FF3B30" }}>
            <XCircle size={16} />
            <span className="font-medium">Model not found</span>
          </div>
        )}
      </section>


    </motion.aside>
  );
}
