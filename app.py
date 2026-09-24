"""
app.py — Accident Detection Demo
Streamlit web app powered by YOLO26m (best.pt)

Run with:
    streamlit run app.py
"""

import os
import io
import time
import tempfile
import subprocess
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "accident_yolo26m" / "weights" / "best.pt"

MODEL_STATS = {
    "Architecture": "YOLO26m",
    "Epochs Trained": 100,
    "Training Images": "8,759",
    "mAP@0.50": "0.179",
    "mAP@0.50:95": "0.092",
    "Precision": "0.707",
    "Recall": "0.246",
    "Parameters": "20.3 M",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS – dark card style, red accent
st.markdown("""
<style>
    /* Header */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #FF3B30;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    /* Detection badge */
    .badge-accident {
        background: #FF3B30;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    .badge-clear {
        background: #34C759;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #eee !important;
    }
    /* Divider */
    hr { border-color: #333; }
</style>
""", unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSmoother:
    """
    Rolling-window multi-frame confirmation (no retraining needed).

    Requires N consecutive frames all above conf threshold before confirming
    an incident. Kills single-frame ghost detections (e.g. close cars) while
    preserving real accidents which persist across many frames.
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self.confirmed_incidents = 0
        self._was_confirmed = False

    def update(self, detected: bool) -> bool:
        """Return True when last N frames ALL had a detection."""
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
# MODEL LOADER (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO26m model…")
def load_model():
    return YOLO(str(MODEL_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def run_on_image(model, img_bgr: np.ndarray, conf: float):
    """Run model on a BGR numpy array. Returns (annotated_bgr, boxes_list)."""
    result = model.predict(img_bgr, conf=conf, verbose=False)[0]
    annotated = result.plot(line_width=2)   # returns BGR with boxes drawn
    boxes = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box, conf_score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box[:4]
            boxes.append({
                "Confidence": f"{conf_score:.2%}",
                "X1": int(x1), "Y1": int(y1),
                "X2": int(x2), "Y2": int(y2),
                "Width (px)": int(x2 - x1),
                "Height (px)": int(y2 - y1),
            })
    return annotated, boxes


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def run_on_video(model, video_path: str, conf: float, smoothing_window: int,
                 progress_bar, status_text):
    """
    Process video frame by frame with temporal smoothing.
    - Raw detections (single frame) shown with YELLOW box
    - Confirmed incidents (N consecutive frames) shown with RED box
    Returns (output_path, stats_dict).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_path = video_path + "_raw.mp4"
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

        result = model.predict(frame, conf=conf, verbose=False)[0]
        has_detection = result.boxes is not None and len(result.boxes) > 0
        is_confirmed  = smoother.update(has_detection)

        # Draw boxes — colour depends on confirmation state
        if has_detection:
            frames_with_raw_detection += 1
            box_color = (0, 0, 255) if is_confirmed else (0, 215, 255)  # Red if confirmed, Yellow if raw
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

        # Overlay status on frame
        status_label = "INCIDENT CONFIRMED" if is_confirmed else ("Detection..." if has_detection else "")
        if status_label:
            cv2.putText(frame, status_label, (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 0, 255) if is_confirmed else (0, 215, 255), 3)

        writer.write(frame)
        frame_idx += 1
        pct = int((frame_idx / max(total_frames, 1)) * 100)
        progress_bar.progress(min(pct, 100))
        status_text.text(
            f"Frame {frame_idx}/{total_frames}  |  "
            f"Raw detections: {frames_with_raw_detection}  |  "
            f"Confirmed incidents: {smoother.confirmed_incidents}"
        )

    cap.release()
    writer.release()

    # Re-encode to H.264 for browser playback
    h264_path = video_path + "_out.mp4"
    status_text.text("Re-encoding to H.264 for browser playback…")
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

    out_path = h264_path if Path(h264_path).exists() else None

    stats = {
        "total_frames": total_frames,
        "raw_detections": frames_with_raw_detection,
        "confirmed_incidents": smoother.confirmed_incidents,
        "smoothing_window": smoothing_window,
        "avg_confidence": f"{np.mean(conf_scores_all):.2%}" if conf_scores_all else "N/A",
        "duration_sec": f"{total_frames/fps:.1f}s",
        "fps": f"{fps:.0f}",
    }
    return out_path, stats




# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.30, max_value=0.90,
        value=0.60, step=0.05,
        help="Only detections above this score are shown. 0.60 recommended.",
    )
    smoothing_window = st.slider(
        "Temporal Smoothing — N consecutive frames",
        min_value=1, max_value=7, value=3, step=1,
        help="Require N consecutive frames with a detection before confirming an incident. N=1 = no smoothing. Only applies to video.",
    )
    if smoothing_window == 1:
        st.caption("ℹ️ Smoothing OFF — every single-frame detection counts")
    else:
        st.caption(f"✅ Smoothing ON — need **{smoothing_window}** consecutive frames to confirm")

    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    for k, v in MODEL_STATS.items():
        st.markdown(f"**{k}:** {v}")

    st.markdown("---")
    st.markdown("### 📂 Weights")
    weight_exists = MODEL_PATH.exists()
    if weight_exists:
        st.success(f"✅ `best.pt` loaded")
    else:
        st.error(f"❌ Model not found at `{MODEL_PATH}`")

    st.markdown("---")
    st.caption("Accident Detection · YOLO26m · College Project Demo")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🚨 Accident Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by YOLO26m · Upload an image or video to detect road accidents</p>', unsafe_allow_html=True)
st.divider()

# Load model
model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_image, tab_video = st.tabs(["🖼️  Image", "🎬  Video"])


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded_img = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="img_uploader",
        help="Supports JPG, JPEG, PNG",
    )

    if uploaded_img is not None:
        # Decode
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode image. Please try a different file.")
        else:
            h, w = img_bgr.shape[:2]

            with st.spinner("Running YOLO26m…"):
                t0 = time.time()
                annotated_bgr, boxes = run_on_image(model, img_bgr, conf_threshold)
                elapsed = time.time() - t0

            # ── Detection Banner ──────────────────────────────────────────
            if boxes:
                st.markdown(
                    f'<span class="badge-accident">🚨 ACCIDENT DETECTED — {len(boxes)} detection{"s" if len(boxes)>1 else ""}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge-clear">✅ No Accident Detected</span>',
                    unsafe_allow_html=True,
                )

            st.caption(f"Inference time: {elapsed*1000:.0f} ms  ·  Image: {w}×{h}px  ·  Conf threshold: {conf_threshold}")
            st.write("")

            # ── Side-by-side ──────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(bgr_to_pil(img_bgr), use_container_width=True)
            with col2:
                st.markdown("**Annotated**")
                annotated_pil = bgr_to_pil(annotated_bgr)
                st.image(annotated_pil, use_container_width=True)

            # ── Download ──────────────────────────────────────────────────
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=pil_to_bytes(annotated_pil),
                file_name=f"accident_detected_{uploaded_img.name}",
                mime="image/jpeg",
            )

            # ── Detection Table ───────────────────────────────────────────
            if boxes:
                st.markdown("### 📋 Detection Details")
                st.dataframe(boxes, use_container_width=True, hide_index=True)
            else:
                st.info("No detections above the confidence threshold. Try lowering the slider in the sidebar.")
    else:
        st.info("👆 Upload an image to get started.")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown(
        "> **Tip:** Keep clips short (< 30 sec) for faster processing on CPU. "
        "Inference runs at ~600 ms/frame on Apple M1."
    )

    uploaded_vid = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="vid_uploader",
        help="Supports MP4, AVI, MOV",
    )

    if uploaded_vid is not None:
        # Write to temp file (OpenCV needs a path)
        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        # Quick preview of source video
        st.markdown("**Source Video**")
        st.video(tmp_path)

        if st.button("▶️ Run Accident Detection", type="primary"):
            st.markdown("---")
            st.markdown("**Processing…**")
            progress_bar = st.progress(0)
            status_text  = st.empty()

            with st.spinner("Analysing video…"):
                out_path, stats = run_on_video(
                    model, tmp_path, conf_threshold, smoothing_window,
                    progress_bar, status_text,
                )

            progress_bar.progress(100)
            status_text.text("✅ Done!")

            # ── Stats Banner ─────────────────────────────────────────────
            st.markdown("### 📊 Detection Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frames", stats["total_frames"])
            c2.metric("Raw Detections", stats["raw_detections"],
                      help="Frames where model fired above conf threshold (before smoothing)")
            c3.metric(f"Confirmed Incidents (N={smoothing_window})", stats["confirmed_incidents"],
                      help=f"Distinct incidents confirmed by {smoothing_window} consecutive frames")
            c4.metric("Avg Confidence", stats["avg_confidence"])

            st.caption(
                f"Duration: {stats['duration_sec']}  ·  FPS: {stats['fps']}  ·  "
                f"Conf: {conf_threshold}  ·  Smoothing N={smoothing_window}"
            )

            if stats["confirmed_incidents"] > 0:
                st.markdown(
                    f'<span class="badge-accident">🚨 {stats["confirmed_incidents"]} confirmed incident(s) — {stats["raw_detections"]} raw detections suppressed to {stats["confirmed_incidents"]} real events</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge-clear">✅ No Accidents Detected</span>',
                    unsafe_allow_html=True,
                )

            # ── Output Video ─────────────────────────────────────────────
            st.markdown("### 🎬 Annotated Output")
            if out_path and Path(out_path).exists():
                st.video(out_path)
                # Download button
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=f.read(),
                        file_name=f"accident_detected_{uploaded_vid.name}",
                        mime="video/mp4",
                    )
            else:
                st.error("Video processing failed. Please try a different file.")

            # Cleanup temp files
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    else:
        st.info("👆 Upload a video to get started.")
