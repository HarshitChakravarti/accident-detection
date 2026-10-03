# Accident Detection Project Context

This document provides a comprehensive overview of the Accident Detection project, including its architecture, tech stack, model details, and evaluation metrics. 

## 1. Project Overview
This project is an **Accident Detection System** powered by a **YOLO26m** object detection model. It is designed to identify vehicle accidents in images and videos. The project features a robust evaluation pipeline, a standalone Streamlit web application, and a modern client-server web architecture using Next.js and FastAPI.

The primary use case is analyzing traffic or dashcam footage to detect real-time accidents while suppressing false positives (ghost detections) using a custom temporal smoothing technique.

---

## 2. Tech Stack

### Machine Learning & Data Science
* **Model Architecture**: YOLO26m (via [Ultralytics](https://github.com/ultralytics/ultralytics))
* **Computer Vision**: OpenCV (`cv2`) for image/video manipulation and bounding box rendering.
* **Evaluation & Metrics**: `scikit-learn` (confusion matrices, classification reports), `matplotlib`, `seaborn`, `pandas`.
* **Video Encoding**: `FFmpeg` (invoked via subprocess) for re-encoding raw OpenCV `.mp4v` outputs to H.264 for web browser compatibility.

### Backend (FastAPI)
* **Framework**: FastAPI (running on Uvicorn)
* **Endpoints**: 
  * `GET /health`: Model loading status and configuration.
  * `POST /detect/image`: Synchronous inference on uploaded images returning base64 annotated images and bounding box coordinates.
  * `POST /detect/video`: Asynchronous video processing utilizing **Server-Sent Events (SSE)** to stream frame-by-frame progress to the client.
  * `GET /detect/video/download/{job_id}`: Video retrieval endpoint.
* **Language**: Python 3.x

### Frontend (Next.js)
* **Framework**: Next.js 16.3.6 (App Router)
* **UI Library**: React 19.2.8
* **Styling**: Tailwind CSS v4, [shadcn/ui](https://ui.shadcn.com/), [Base UI](https://base-ui.com/).
* **Animation**: Framer Motion (`framer-motion`)
* **Icons**: Lucide React
* **Client-Server Comm**: `fetch` API handling both multipart form data and parsing Server-Sent Events (SSE) streams for video inference progress bars.

### Alternative / Legacy UI
* **Streamlit**: A fully functional, standalone `app.py` script providing a fast Python-only GUI for both image and video inference without needing the Next.js/FastAPI split.

---

## 3. Dataset Information
* **Format**: Standard YOLO format (`images/train`, `images/val`, `images/test` and corresponding `labels/`).
* **Classes**: A single target class (`nc: 1`), labeled as **`'Accident'`**.
* **Size**: 8,759 training images.
* **Configuration File**: `data.yaml`

---

## 4. Model & Training Details
The model weights are located in `accident_yolo26m/weights/best.pt`.

* **Base Architecture**: YOLO26m
* **Parameters**: 20.3 Million
* **Epochs Trained**: 100
* **Batch Size**: 16
* **Image Size (imgsz)**: 640x640
* **Optimizer Setup**: Initial LR `0.01`, Momentum `0.937`, Weight Decay `0.0005`.

---

## 5. Model Evaluation Metrics
Based on the `evaluate.py` pipeline and official model statistics:

* **mAP@0.50**: 0.179
* **mAP@0.50:95**: 0.092
* **Precision**: 0.707
* **Recall**: 0.246

The custom `evaluate.py` script performs:
1. Standard YOLO validation metric calculation (mAP).
2. Per-image inference to generate image-level confusion matrices (Binary: Accident vs. No Accident).
3. Box-level Precision, Recall, and F1 Score calculations (using an IoU threshold of 0.50).
4. Plotting of training curves (Box Loss, Class Loss, mAP) parsed directly from the `results.csv` logs.
5. Generation of an annotated image grid comparing ground-truth boxes (Green) vs. predicted boxes (Red).

---

## 6. Temporal Smoothing Technique
A major challenge with per-frame object detection on video is "flickering" or single-frame false positives (e.g., cars driving closely together temporarily looking like a crash). 

To solve this, the project implements a **Rolling-Window Temporal Smoother** (`TemporalSmoother` class).

### How it works:
* A rolling buffer (deque) of size `N` (default window size is 3 frames) tracks the boolean detection state of the last `N` frames.
* **Raw Detection**: The YOLO model detects an accident in a single frame. (Highlighted in Yellow bounding boxes).
* **Confirmed Incident**: An incident is only "Confirmed" (and counted towards the final metric) if the model detects an accident in **all `N` consecutive frames** within the window. (Highlighted in Red bounding boxes).
* **Impact**: It effectively kills 1-frame ghost detections while preserving real accidents, which typically persist across many consecutive frames. This provides a massive improvement in practical video inference without requiring recurrent neural networks (RNNs) or model retraining.
