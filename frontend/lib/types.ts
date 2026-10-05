// lib/types.ts — shared TypeScript types for the Accident Detection app

export interface BoundingBox {
  confidence: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
}

export interface ImageDetectionResult {
  annotated_image: string;  // base64 JPEG
  original_image: string;   // base64 JPEG
  boxes: BoundingBox[];
  elapsed_ms: number;
  image_size: { width: number; height: number };
  conf_threshold: number;
}

export interface VideoProgress {
  type: "progress";
  progress: number;        // 0–99
  frame: number;
  totalFrames: number;
  rawDetections: number;
  confirmedIncidents: number;
}

export interface VideoStats {
  totalFrames: number;
  rawDetections: number;
  confirmedIncidents: number;
  smoothingWindow: number;
  avgConfidence: string;
  durationSec: string;
  fps: string;
  videoAvailable: boolean;
}

export interface VideoCompleteEvent {
  type: "complete";
  jobId: string;
  stats: VideoStats;
}

export interface VideoErrorEvent {
  type: "error";
  message: string;
}

export type VideoSSEEvent = VideoProgress | VideoCompleteEvent | VideoErrorEvent;

export interface ModelStats {
  Architecture: string;
  "Epochs Trained": number;
  "Training Images": string;
  "mAP@0.50": string;
  "mAP@0.50:95": string;
  Precision: string;
  Recall: string;
  Parameters: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
  model_stats: ModelStats;
}
