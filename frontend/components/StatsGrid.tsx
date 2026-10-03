"use client";

import { motion } from "framer-motion";

interface StatItem {
  label: string;
  value: string | number;
  hint?: string;
  accent?: boolean;
}

interface StatsGridProps {
  stats: StatItem[];
}

export default function StatsGrid({ stats }: StatsGridProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {stats.map((s, i) => (
        <motion.div
          key={s.label}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06, duration: 0.3 }}
          className="glass-card p-4 flex flex-col gap-1"
          style={{
            boxShadow: s.accent && Number(s.value) > 0
              ? "0 0 20px rgba(255,59,48,0.12)"
              : undefined,
          }}
        >
          <p className="text-xs font-medium uppercase tracking-wide" style={{ color: "#666" }}>
            {s.label}
          </p>
          <p
            className="text-2xl font-bold tabular-nums"
            style={{ color: s.accent && Number(s.value) > 0 ? "#FF3B30" : "#f0f0f0" }}
          >
            {s.value}
          </p>
          {s.hint && (
            <p className="text-xs leading-tight" style={{ color: "#555" }}>
              {s.hint}
            </p>
          )}
        </motion.div>
      ))}
    </div>
  );
}
