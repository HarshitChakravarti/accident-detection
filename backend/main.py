"""
backend/main.py — FastAPI inference server for Accident Detection
Replaces Streamlit as the serving layer; the Next.js frontend calls this API.

Run with:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import uuid
from collections import deque
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # repo root
MODEL_PATH = BASE_DIR / "accident_yolo26m" / "weights" / "best.pt"
TMP_DIR    = Path(tempfile.gettempdir()) / "accident_detection_jobs"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STATS = {
    "Architecture":     "YOLO26m",
    "Epochs Trained":   100,
    "Training Images":  "8,759",
    "mAP@0.50":         "0.179",
    "mAP@0.50:95":      "0.092",
    "Precision":        "0.707",
    "Recall":           "0.426",
    "Parameters":       "20.3 M",
}

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Accident Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Determine best device (MPS for Apple Silicon, else CPU)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading Accident YOLO model from {MODEL_PATH} on {DEVICE.upper()}…")
try:
    model = YOLO(str(MODEL_PATH))
    model_loaded = True
    print("Accident Model loaded ✅")
except Exception as exc:
    model = None
    model_loaded = False
    print(f"Failed to load Accident Model: {exc}")

print(f"Loading Tracker YOLO model (yolov8n.pt) on {DEVICE.upper()}…")
try:
    tracker_model = YOLO("yolov8n.pt")
    print("Tracker Model loaded ✅")
except Exception as exc:
    tracker_model = None
    print(f"Failed to load Tracker Model: {exc}")



# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHER (same as app.py)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSmoother:
    """Rolling-window multi-frame confirmation."""

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self.confirmed_incidents = 0
        self._was_confirmed = False

    def update(self, detected: bool) -> bool:
        self._window.append(1 if detected else 0)
        is_confirmed = (
            len(self._window) == self.window_size
            and sum(self._window) == self.window_size
        )
        if is_confirmed and not self._was_confirmed:
            self.confirmed_incidents += 1
        self._was_confirmed = is_confirmed
        return is_confirmed

    def reset(self):
        self._window.clear()
        self._was_confirmed = False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _require_model():
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH.")


def _bgr_to_jpeg_b64(bgr: np.ndarray) -> str:
    """Convert BGR numpy array → base64-encoded JPEG string."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _run_on_image(img_bgr: np.ndarray, conf: float):
    """Run inference on a BGR image. Returns (annotated_b64, boxes_list)."""
    annotated = img_bgr.copy()

    if tracker_model is not None:
        tr_res = tracker_model.predict(img_bgr, classes=[2, 3, 5, 7], device=DEVICE, verbose=False)[0]
        annotated = tr_res.plot(img=annotated, line_width=1)

    result = model.predict(img_bgr, conf=conf, device=DEVICE, verbose=False)[0]
    annotated = result.plot(img=annotated, line_width=2)
    boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, conf_score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box[:4]
            boxes.append({
                "confidence": f"{conf_score:.2%}",
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "width":  int(x2 - x1),
                "height": int(y2 - y1),
            })
    return _bgr_to_jpeg_b64(annotated), boxes


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if model_loaded else "model_missing",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "model_stats": MODEL_STATS,
    }


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.60),
):
    """
    Accepts an image upload and returns:
      - annotated_image: base64 JPEG
      - original_image:  base64 JPEG
      - boxes: list of bounding box dicts
      - elapsed_ms: inference duration
      - image_size: { width, height }
    """
    _require_model()

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    h, w = img_bgr.shape[:2]

    t0 = time.time()
    annotated_b64, boxes = _run_on_image(img_bgr, conf)
    elapsed_ms = int((time.time() - t0) * 1000)

    original_b64 = _bgr_to_jpeg_b64(img_bgr)

    return {
        "annotated_image": annotated_b64,
        "original_image":  original_b64,
        "boxes":           boxes,
        "elapsed_ms":      elapsed_ms,
        "image_size":      {"width": w, "height": h},
        "conf_threshold":  conf,
    }


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(0.60),
    smoothing: int = Form(3),
):
    """
    Accepts a video upload and streams Server-Sent Events (SSE) with progress.
    
    SSE event types:
      - progress:  { progress, frame, totalFrames, rawDetections, confirmedIncidents }
      - complete:  { jobId, stats }
      - error:     { message }
    
    After receiving `complete`, call GET /detect/video/download/{jobId} for the video.
    """
    _require_model()

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    job_id = str(uuid.uuid4())

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    async def event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        try:
            # Run video processing in a thread pool (CPU-bound)
            result = await loop.run_in_executor(
                None,
                _process_video_sync,
                tmp_path, conf, smoothing, job_id,
                lambda event_data: asyncio.run_coroutine_threadsafe(
                    _send_sse_queue.put(event_data), loop
                )
            )

            # Yield all queued events
            while not _send_sse_queue.empty():
                yield _send_sse_queue.get_nowait()

            yield f"data: {json.dumps({'type': 'complete', **result})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # Use a simpler generator-based approach
    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        async def run_processing():
            try:
                await loop.run_in_executor(
                    None,
                    lambda: _process_video_sync_with_queue(
                        tmp_path, conf, smoothing, job_id, queue, loop
                    )
                )
            except Exception as exc:
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)  # sentinel

        task = asyncio.create_task(run_processing())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0)  # allow event loop to breathe

        await task

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _process_video_sync_with_queue(
    video_path: str,
    conf: float,
    smoothing_window: int,
    job_id: str,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
):
    """
    CPU-bound video processing. Posts progress events to asyncio queue.
    Returns final stats dict as a 'complete' event.
    """
    def put(data: dict):
        asyncio.run_coroutine_threadsafe(queue.put(data), loop).result(timeout=5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        put({"type": "error", "message": "Could not open video file."})
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_path = str(TMP_DIR / f"{job_id}_raw.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    smoother = TemporalSmoother(window_size=smoothing_window)
    frames_with_raw_detection = 0
    conf_scores_all = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 1. Vehicle Tracking ---
        if tracker_model is not None:
            # classes: 2=car, 3=motorcycle, 5=bus, 7=truck
            tr_results = tracker_model.track(frame, persist=True, classes=[2, 3, 5, 7], device=DEVICE, verbose=False)
            if tr_results and tr_results[0].boxes is not None and tr_results[0].boxes.id is not None:
                boxes_tr = tr_results[0].boxes.xyxy.cpu().numpy()
                track_ids = tr_results[0].boxes.id.int().cpu().tolist()
                for box, tid in zip(boxes_tr, track_ids):
                    tx1, ty1, tx2, ty2 = map(int, box[:4])
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (100, 255, 100), 1)
                    cv2.putText(frame, f"ID: {tid}", (tx1, max(ty1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        # --- 2. Accident Detection ---
        result = model.predict(frame, conf=conf, device=DEVICE, verbose=False)[0]
        has_detection = result.boxes is not None and len(result.boxes) > 0
        is_confirmed  = smoother.update(has_detection)

        if has_detection:
            frames_with_raw_detection += 1
            box_color = (0, 0, 255) if is_confirmed else (0, 215, 255)
            for box, conf_score in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = f"{'CONFIRMED' if is_confirmed else 'RAW'} {conf_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
                conf_scores_all.append(float(conf_score))

        status_label = "INCIDENT CONFIRMED" if is_confirmed else ("Detection..." if has_detection else "")
        if status_label:
            cv2.putText(frame, status_label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 255) if is_confirmed else (0, 215, 255), 3)

        writer.write(frame)
        frame_idx += 1

        pct = int((frame_idx / max(total_frames, 1)) * 100)
        # Send progress every 5 frames (or always if total < 50)
        if frame_idx % max(1, total_frames // 20) == 0 or frame_idx == total_frames:
            put({
                "type": "progress",
                "progress": min(pct, 99),
                "frame": frame_idx,
                "totalFrames": total_frames,
                "rawDetections": frames_with_raw_detection,
                "confirmedIncidents": smoother.confirmed_incidents,
            })

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser playback
    h264_path = str(TMP_DIR / f"{job_id}_out.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", h264_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        os.unlink(raw_path)
    except Exception:
        pass

    video_exists = Path(h264_path).exists()
    avg_conf = f"{np.mean(conf_scores_all):.2%}" if conf_scores_all else "N/A"
    duration_sec = f"{total_frames / fps:.1f}s"

    stats = {
        "totalFrames":        total_frames,
        "rawDetections":      frames_with_raw_detection,
        "confirmedIncidents": smoother.confirmed_incidents,
        "smoothingWindow":    smoothing_window,
        "avgConfidence":      avg_conf,
        "durationSec":        duration_sec,
        "fps":                f"{fps:.0f}",
        "videoAvailable":     video_exists,
    }

    put({
        "type":   "complete",
        "jobId":  job_id,
        "stats":  stats,
    })


@app.get("/detect/video/download/{job_id}")
async def download_video(job_id: str):
    """Return the processed video file for a completed job."""
    # Sanitise job_id (UUID only)
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID.")

    out_path = TMP_DIR / f"{job_id}_out.mp4"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Video not found or expired.")

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=f"accident_detected_{job_id}.mp4",
    )
