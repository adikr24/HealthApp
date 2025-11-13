#!/usr/bin/env python3
# NOTE: keeps original behavior (saves annotated images) and ALSO writes a CSV of detections.

## this code generates only person annotations from a detectron2 model and will be clubbed with custom yolo for merged predictions ##

from projectcode.ML.StandardClasses.RunYoloOnVideoOrImages import YoloRelevantAnnotator
from pathlib import Path

# === NEW: for CSV export ===
import csv, cv2
from ultralytics import YOLO
# ===========================

#### CONFIG ##############################################################
# Path to your trained YOLO model
WEIGHTS = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloResultIterations/IterationIIResults/union93_yolov8n_6403/weights/best.pt"

# Folder with extracted frames from your input video
FRAMES_DIR = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeIII/ExtractFrames/ProteinShakeInf")

# Output folder for annotated results
OUT_DIR_YOLO = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo")

# Inference parameters
EVERY_NTH  = 1          # analyze every 2nd frame
CONF_THRES = 0.5        # confidence threshold (≥ 50%)
MAX_FRAMES = 2200       # cap total frames processed

# === NEW: CSV output path ===
SUMMARY_CSV = OUT_DIR_YOLO / "custom_yolo_detections.csv"
# ============================

#######################################################################
# Ensure output directory exists
OUT_DIR_YOLO.mkdir(parents=True, exist_ok=True)

# Sanity checks
if not Path(WEIGHTS).exists():
    print(f"[WARN] YOLO weights not found: {WEIGHTS}")
if not FRAMES_DIR.exists():
    print(f"[WARN] Frames directory not found: {FRAMES_DIR}")

# Collect frames
all_frames = sorted(list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.png")))
sampled_frames = all_frames[::EVERY_NTH][:MAX_FRAMES]

print(f"[INFO] Total frames found: {len(all_frames)}")
print(f"[INFO] Sampled (every {EVERY_NTH}, max {MAX_FRAMES}): {len(sampled_frames)}")
if sampled_frames:
    print(f"[INFO] First sampled frame: {sampled_frames[0]}")

#######################################################################
# Initialize YOLO annotator (DRAWING stays as-is)
annot = YoloRelevantAnnotator(
    model_weights=WEIGHTS,
    conf_threshold=CONF_THRES,
    relevant_labels_csv="/app/mediaFiles/yolo_settings/YoloLabels/CustomIterationLabel_I.csv",
)

# === NEW: set up Ultralytics model for CSV extraction (separate from annotator) ===
yolo_for_csv = YOLO(str(WEIGHTS))
id2name = yolo_for_csv.model.names if hasattr(yolo_for_csv.model, "names") else {}
# Prepare CSV header
csv_rows = [(
    "frame","img_w","img_h",
    "cls_id","cls_name","conf",
    "x1","y1","x2","y2",
    "cx_norm","cy_norm","w_norm","h_norm"
)]
# =====================================================================

#######################################################################
# Run inference
saved_yolo = 0
for i, fp in enumerate(sampled_frames, start=1):
    out_path = OUT_DIR_YOLO / fp.name
    has_rel = annot.process_image_file(fp, out_path)   # <-- keeps your original annotation behavior
    status = "✓" if has_rel else "-"
    print(f"[{i:03d}/{len(sampled_frames)}] {fp.name} -> {status}")
    if has_rel:
        saved_yolo += 1

    # === NEW: append detections to CSV for this frame ===
    img = cv2.imread(str(fp))
    if img is None:
        print(f"[WARN] Skipping unreadable frame for CSV: {fp}")
        continue
    H, W = img.shape[:2]

    res = yolo_for_csv.predict(source=img, conf=CONF_THRES, iou=0.45, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        # Optional: still log that the frame was processed (comment out if you want a sparse CSV)
        # csv_rows.append((fp.name, W, H, "", "", "", "", "", "", "", "", "", "", ""))
        continue

    for b in res.boxes:
        cls_id = int(b.cls.item())
        conf   = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        cx = (x1 + x2) / (2 * W); cy = (y1 + y2) / (2 * H)
        wN = (x2 - x1) / W;       hN = (y2 - y1) / H
        csv_rows.append((
            fp.name, W, H,
            cls_id, id2name.get(cls_id, f"cls_{cls_id}"), f"{conf:.3f}",
            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            f"{cx:.4f}", f"{cy:.4f}", f"{wN:.4f}", f"{hN:.4f}"
        ))
    # === END NEW CSV block ===

#######################################################################
# === NEW: write the CSV to disk ===
with open(SUMMARY_CSV, "w", newline="") as f:
    csv.writer(f).writerows(csv_rows)
# ==================================

print(f"\n[YOLO] Frames with detections saved: {saved_yolo}/{len(sampled_frames)}")
print(f"[YOLO] Output directory: {OUT_DIR_YOLO}")
print(f"[YOLO] Summary CSV: {SUMMARY_CSV}")
print("[DONE] Inference complete. Review annotated images and CSV above.")
