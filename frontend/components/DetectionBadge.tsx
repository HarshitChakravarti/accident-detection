"use client";

import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle } from "lucide-react";

interface DetectionBadgeProps {
  variant: "accident" | "clear";
  label: string;
}

export default function DetectionBadge({ variant, label }: DetectionBadgeProps) {
  const isAccident = variant === "accident";

  return (
    <motion.div
      initial={{ scale: 0.6, opacity: 0, y: 8 }}
      animate={{ scale: 1, opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 500, damping: 28 }}
      className="inline-flex items-center gap-2 px-4 py-2 rounded-full font-bold text-sm"
      style={{
        background: isAccident ? "rgba(255,59,48,0.15)" : "rgba(52,199,89,0.12)",
        color: isAccident ? "#FF3B30" : "#34C759",
        border: `1px solid ${isAccident ? "rgba(255,59,48,0.4)" : "rgba(52,199,89,0.35)"}`,
        boxShadow: isAccident
          ? "0 0 20px rgba(255,59,48,0.2)"
          : "0 0 20px rgba(52,199,89,0.15)",
      }}
    >
      {isAccident ? <AlertTriangle size={16} /> : <CheckCircle size={16} />}
      {label}
    </motion.div>
  );
}
