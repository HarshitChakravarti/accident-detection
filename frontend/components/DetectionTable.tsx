"use client";

import { motion } from "framer-motion";
import type { BoundingBox } from "@/lib/types";

interface DetectionTableProps {
  boxes: BoundingBox[];
}

export default function DetectionTable({ boxes }: DetectionTableProps) {
  if (boxes.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="glass-card overflow-hidden"
    >
      <div className="px-5 py-4 border-b" style={{ borderColor: "rgba(255,255,255,0.07)" }}>
        <h3 className="font-semibold text-sm" style={{ color: "#f0f0f0" }}>
          📋 Detection Details
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              {["#", "Confidence", "X1", "Y1", "X2", "Y2", "W×H"].map((h) => (
                <th
                  key={h}
                  className="px-4 py-3 text-left font-medium tabular-nums"
                  style={{ color: "#666" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {boxes.map((b, i) => (
              <motion.tr
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="transition-colors"
                style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                }}
              >
                <td className="px-4 py-3 font-mono" style={{ color: "#555" }}>
                  {i + 1}
                </td>
                <td className="px-4 py-3">
                  <span
                    className="px-2 py-0.5 rounded-md font-bold text-xs"
                    style={{ background: "rgba(255,59,48,0.12)", color: "#FF3B30" }}
                  >
                    {b.confidence}
                  </span>
                </td>
                <td className="px-4 py-3 font-mono text-white/70">{b.x1}</td>
                <td className="px-4 py-3 font-mono text-white/70">{b.y1}</td>
                <td className="px-4 py-3 font-mono text-white/70">{b.x2}</td>
                <td className="px-4 py-3 font-mono text-white/70">{b.y2}</td>
                <td className="px-4 py-3 font-mono text-white/70">
                  {b.width}×{b.height}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
