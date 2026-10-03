"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Sidebar from "@/components/Sidebar";
import ImageTab from "@/components/ImageTab";
import VideoTab from "@/components/VideoTab";

type Tab = "image" | "video";

export default function Home() {
  const [conf, setConf] = useState(0.60);
  const [smoothing, setSmoothing] = useState(3);
  const [tab, setTab] = useState<Tab>("image");

  return (
    <div className="flex min-h-screen" style={{ background: "#0a0a0f" }}>
      {/* Sidebar */}
      <Sidebar
        conf={conf}
        setConf={setConf}
        smoothing={smoothing}
        setSmoothing={setSmoothing}
      />

      {/* Main content */}
      <main className="flex-1 min-w-0 flex flex-col">
        {/* Topbar */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="sticky top-0 z-10 px-8 py-5 flex items-end justify-between"
          style={{
            background: "rgba(10,10,15,0.8)",
            backdropFilter: "blur(12px)",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div>
            <h1
              className="text-3xl font-extrabold tracking-tight"
              style={{
                color: "#f0f0f0",
                textShadow: "0 0 40px rgba(255,59,48,0.15)",
              }}
            >
              Accident Detection
            </h1>
            <p className="text-sm mt-0.5" style={{ color: "#555" }}>
              YOLO26m · Upload an image or video to detect road accidents
            </p>
          </div>

          {/* Tab switcher */}
          <div
            className="flex rounded-xl p-1"
            style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.07)" }}
          >
            {(["image", "video"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="relative px-5 py-2 rounded-lg text-sm font-medium capitalize transition-colors"
                style={{ color: tab === t ? "#f0f0f0" : "#666" }}
              >
                {tab === t && (
                  <motion.div
                    layoutId="tab-indicator"
                    className="absolute inset-0 rounded-lg"
                    style={{ background: "rgba(255,59,48,0.15)", border: "1px solid rgba(255,59,48,0.25)" }}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
                <span className="relative z-10">
                  {t === "image" ? "🖼️ Image" : "🎬 Video"}
                </span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Tab content */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="flex-1 p-8"
        >
          {tab === "image" ? (
            <ImageTab conf={conf} />
          ) : (
            <VideoTab conf={conf} smoothing={smoothing} />
          )}
        </motion.div>
      </main>
    </div>
  );
}
