"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { Upload, ImageIcon, VideoIcon } from "lucide-react";

interface DropZoneProps {
  accept: "image" | "video";
  onFile: (file: File) => void;
  disabled?: boolean;
  label?: string;
}

const IMAGE_TYPES = { "image/*": [".jpg", ".jpeg", ".png"] };
const VIDEO_TYPES = { "video/*": [".mp4", ".avi", ".mov"] };

export default function DropZone({ accept, onFile, disabled, label }: DropZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFile(accepted[0]);
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept === "image" ? IMAGE_TYPES : VIDEO_TYPES,
    multiple: false,
    disabled,
  });

  const Icon = accept === "image" ? ImageIcon : VideoIcon;
  const ext = accept === "image" ? ".jpg / .jpeg / .png" : ".mp4 / .avi / .mov";
  const defaultLabel = label ?? (accept === "image" ? "Drop an image here" : "Drop a video here");

  // Use a plain div for the dropzone root to avoid framer-motion onDrag type conflict
  const rootProps = getRootProps();

  return (
    <div
      {...rootProps}
      className={[
        "relative flex flex-col items-center justify-center gap-4",
        "p-10 rounded-2xl border-2 border-dashed cursor-pointer",
        "transition-all duration-200 select-none",
        disabled ? "opacity-50 cursor-not-allowed" : "",
        isDragActive
          ? "border-[#FF3B30] bg-[rgba(255,59,48,0.07)]"
          : "border-white/10 hover:border-white/25 bg-white/[0.02] hover:bg-white/[0.04]",
      ].join(" ")}
      style={{
        boxShadow: isDragActive
          ? "0 0 40px rgba(255,59,48,0.15), inset 0 0 30px rgba(255,59,48,0.04)"
          : "none",
      }}
    >
      <input {...getInputProps()} />

      <motion.div
        animate={{ scale: isDragActive ? 1.2 : 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 20 }}
        className="flex items-center justify-center w-14 h-14 rounded-2xl"
        style={{
          background: isDragActive ? "rgba(255,59,48,0.15)" : "rgba(255,255,255,0.06)",
          color: isDragActive ? "#FF3B30" : "rgba(255,255,255,0.4)",
        }}
      >
        {isDragActive ? <Upload size={28} /> : <Icon size={28} />}
      </motion.div>

      <div className="text-center">
        <p
          className="text-sm font-medium"
          style={{ color: isDragActive ? "#FF3B30" : "rgba(255,255,255,0.6)" }}
        >
          {isDragActive ? "Release to upload" : defaultLabel}
        </p>
        <p className="text-xs mt-1" style={{ color: "#555" }}>
          {isDragActive ? "" : `or click to browse · ${ext}`}
        </p>
      </div>
    </div>
  );
}
