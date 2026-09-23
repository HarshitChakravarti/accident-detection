"""
evaluate.py
===========
Full evaluation pipeline for the Accident Detection YOLO26m model.

Steps:
1. Run YOLO val() on the test split → mAP50, mAP50-95, precision, recall
2. Run per-image inference on the test set → collect raw predictions + ground-truth
3. Build a confusion matrix (binary: accident detected vs not)
4. Save annotated prediction images (sample grid)
5. Plot training curves from results.csv
6. Print a clean summary report

Usage:
    python evaluate.py

Outputs are saved to: ./evaluation_results/
"""

import os
import csv
import json
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
)
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "accident_yolo26m" / "weights" / "best.pt"
DATA_YAML    = BASE_DIR / "data.yaml"
TEST_IMG_DIR = BASE_DIR / "yolo_dataset" / "images" / "test"
TEST_LBL_DIR = BASE_DIR / "yolo_dataset" / "labels" / "test"
RESULTS_CSV  = BASE_DIR / "accident_yolo26m" / "results.csv"
OUT_DIR      = BASE_DIR / "evaluation_results_conf60"

CONF_THRESHOLD   = 0.60   # prediction confidence threshold (raised to suppress low-confidence ghost detections)
IOU_THRESHOLD    = 0.50   # IoU for TP matching
NUM_SAMPLE_IMGS  = 20     # how many annotated images to save in the grid
RANDOM_SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def xywhn_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised box to absolute pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path, img_w, img_h):
    """Return list of absolute [x1,y1,x2,y2] ground-truth boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append(xywhn_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def match_predictions(pred_boxes, gt_boxes, iou_thresh=IOU_THRESHOLD):
    """
    Returns (TP, FP, FN) for a single image.
    pred_boxes / gt_boxes: list of [x1,y1,x2,y2].
    """
    matched_gt = set()
    TP = FP = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_idx = i
        if best_iou >= iou_thresh:
            TP += 1
            matched_gt.add(best_idx)
        else:
            FP += 1
    FN = len(gt_boxes) - len(matched_gt)
    return TP, FP, FN


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_img_dir = OUT_DIR / "annotated_predictions"
    pred_img_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Accident Detection – Evaluation Pipeline")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────
    print(f"\n[1/5] Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # ── 2. YOLO official val() on test split ─────────────────────────
    print(f"\n[2/5] Running YOLO val() on test split …")
    val_results = model.val(
        data=str(DATA_YAML),
        split="test",
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False,
        save_json=False,
        plots=True,
        project=str(OUT_DIR),
        name="yolo_val",
        exist_ok=True,
    )

    map50     = float(val_results.box.map50)
    map50_95  = float(val_results.box.map)
    prec      = float(val_results.box.mp)
    rec       = float(val_results.box.mr)

    print(f"   mAP@0.50      : {map50:.4f}")
    print(f"   mAP@0.50:0.95 : {map50_95:.4f}")
    print(f"   Precision     : {prec:.4f}")
    print(f"   Recall        : {rec:.4f}")

    # ── 3. Per-image inference for confusion matrix ───────────────────
    print(f"\n[3/5] Running per-image inference for confusion matrix …")
    test_images = sorted(TEST_IMG_DIR.glob("*.jpg"))
    print(f"   Found {len(test_images)} test images.")

    y_true, y_pred = [], []   # binary: 1 = accident present, 0 = none
    total_tp = total_fp = total_fn = 0

    detailed_rows = []   # for the per-image CSV

    for img_path in test_images:
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes  = load_gt_boxes(lbl_path, w, h)
        has_gt    = len(gt_boxes) > 0

        result     = model.predict(str(img_path), conf=CONF_THRESHOLD,
                                   verbose=False)[0]
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes.xyxy.cpu().numpy():
                pred_boxes.append(box[:4].tolist())
        has_pred = len(pred_boxes) > 0

        TP, FP, FN = match_predictions(pred_boxes, gt_boxes)
        total_tp += TP; total_fp += FP; total_fn += FN

        y_true.append(1 if has_gt   else 0)
        y_pred.append(1 if has_pred else 0)

        detailed_rows.append({
            "image": img_path.name,
            "gt_boxes": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "TP": TP, "FP": FP, "FN": FN,
        })

    # ── 4. Confusion matrix ──────────────────────────────────────────
    print(f"\n[4/5] Building confusion matrix …")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   Confusion Matrix (binary image-level):\n{cm}")

    cls_report = classification_report(y_true, y_pred,
                                       target_names=["No Accident", "Accident"],
                                       zero_division=0)
    print(f"\n   Classification Report:\n{cls_report}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Accident", "Accident"],
                yticklabels=["No Accident", "Accident"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Image-level, Test Set)")
    plt.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {cm_path}")

    # Box-level metrics
    precision_box = total_tp / (total_tp + total_fp + 1e-8)
    recall_box    = total_tp / (total_tp + total_fn + 1e-8)
    f1_box        = 2 * precision_box * recall_box / (precision_box + recall_box + 1e-8)
    print(f"\n   Box-level metrics (IoU≥{IOU_THRESHOLD}):")
    print(f"   Precision : {precision_box:.4f}")
    print(f"   Recall    : {recall_box:.4f}")
    print(f"   F1 Score  : {f1_box:.4f}")
    print(f"   TP={total_tp}  FP={total_fp}  FN={total_fn}")

    # ── 5. Training curves from results.csv ─────────────────────────
    print(f"\n[5/5] Plotting training curves …")
    df = pd.read_csv(RESULTS_CSV)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("YOLO26m Accident Detection – Training Curves", fontsize=14, fontweight="bold")

    # Loss curves
    axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="steelblue")
    axes[0, 0].plot(df["epoch"], df["val/box_loss"],   label="Val Box Loss",   color="coral", linestyle="--")
    axes[0, 0].set_title("Box Loss"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", color="steelblue")
    axes[0, 1].plot(df["epoch"], df["val/cls_loss"],   label="Val Cls Loss",   color="coral", linestyle="--")
    axes[0, 1].set_title("Classification Loss"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # mAP curves
    axes[1, 0].plot(df["epoch"], df["metrics/mAP50(B)"],    label="mAP@0.50",    color="green")
    axes[1, 0].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.50:95", color="darkgreen", linestyle="--")
    axes[1, 0].set_title("mAP"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("mAP"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="purple")
    axes[1, 1].plot(df["epoch"], df["metrics/recall(B)"],    label="Recall",    color="darkorange", linestyle="--")
    axes[1, 1].set_title("Precision & Recall"); axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Value"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = OUT_DIR / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"   Saved → {curves_path}")

    # ── 6. Sample annotated prediction images grid ───────────────────
    print(f"\n   Generating annotated prediction image grid …")
    sample_paths = random.sample(test_images, min(NUM_SAMPLE_IMGS, len(test_images)))

    cols = 4
    rows = (len(sample_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for idx, img_path in enumerate(sample_paths):
        ax = axes[idx]
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        img_bgr  = cv2.imread(str(img_path))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w     = img_rgb.shape[:2]

        result = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]

        ax.imshow(img_rgb)

        # Ground-truth boxes (green)
        for gb in load_gt_boxes(lbl_path, w, h):
            x1, y1, x2, y2 = gb
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

        # Predicted boxes (red)
        if result.boxes is not None and len(result.boxes) > 0:
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box[:4]
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          linewidth=2, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f"{conf:.2f}", color="red",
                        fontsize=7, fontweight="bold")

        ax.set_title(img_path.name[:30], fontsize=6)
        ax.axis("off")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="red",  lw=2, label="Prediction"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle("Accident Detection – Sample Predictions (Red=Pred, Green=GT)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    grid_path = OUT_DIR / "sample_predictions_grid.png"
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"   Saved → {grid_path}")

    # ── 7. Per-image CSV ─────────────────────────────────────────────
    csv_path = OUT_DIR / "per_image_results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["image", "gt_boxes", "pred_boxes", "TP", "FP", "FN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(f"   Per-image CSV → {csv_path}")

    # ── 8. Summary report ────────────────────────────────────────────
    report_path = OUT_DIR / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  ACCIDENT DETECTION – EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model     : {MODEL_PATH}\n")
        f.write(f"Test set  : {TEST_IMG_DIR}\n")
        f.write(f"Images    : {len(test_images)}\n\n")
        f.write("── YOLO val() Metrics (test split) ──\n")
        f.write(f"  mAP@0.50      : {map50:.4f}\n")
        f.write(f"  mAP@0.50:0.95 : {map50_95:.4f}\n")
        f.write(f"  Precision     : {prec:.4f}\n")
        f.write(f"  Recall        : {rec:.4f}\n\n")
        f.write("── Box-level Metrics (IoU ≥ 0.50) ──\n")
        f.write(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}\n")
        f.write(f"  Precision : {precision_box:.4f}\n")
        f.write(f"  Recall    : {recall_box:.4f}\n")
        f.write(f"  F1 Score  : {f1_box:.4f}\n\n")
        f.write("── Image-level Classification Report ──\n")
        f.write(cls_report + "\n")
        f.write("── Output Files ──\n")
        for p in [cm_path, curves_path, grid_path, csv_path]:
            f.write(f"  {p}\n")

    print(f"\n   Full report → {report_path}")
    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
