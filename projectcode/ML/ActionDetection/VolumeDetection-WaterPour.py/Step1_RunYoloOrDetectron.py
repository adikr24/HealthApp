## Bounding boxes for cup detection and flow ROI computation ##
## Step 1 -- Draw bounding boxes around detected cups and compute flow ROIs ##


import os
import glob
import csv
from pathlib import Path

import cv2
import numpy as np
# ======= YOLO (Ultralytics) =======
# pip install ultralytics
from ultralytics import YOLO

# COCO model (has 'cup' class)

# ======= Paths (edit if needed) =======
INPUT_DIR  = "/app/mediaFiles/videos/AnnotateVideos/RawVideos/WaterPour"
OUTPUT_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/water_volume_output/yolopass"
GLOB_PAT   = "frame_*.jpg"   # your frame naming pattern
CONF_THRESH = 0.35           # YOLO confidence threshold for 'cup'
MODEL_WEIGHTS = "/app/mediaFiles/yolo_results/YoloWeights/yolov8n.pt"   # tiny/fast; switch to yolov8s.pt for better accuracy


# ROI shape relative to the detected cup box
ROI_HEIGHT_FRAC = 0.40       # ROI height as a fraction of cup height
ROI_WIDTH_FRAC  = 0.60       # ROI width  as a fraction of cup width
RIM_GAP_FRAC    = 0.03       # small vertical gap above rim (fraction of cup height)

RIGHT_INSET_FRAC = 0.18      # shave this fraction of *cup width* from ROI's right side
MIN_ROI_W = 4                # don't let ROI get too thin
MIN_ROI_H = 4

# Save clean vs. annotated outputs separately
WRITE_ANNOTATIONS = True
ANN_DIR   = os.path.join(OUTPUT_DIR, "annotated_frames")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean_frames")
MASK_DIR  = os.path.join(OUTPUT_DIR, "roi_masks")  # optional: binary ROI mask per frame
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)


def clip(val, lo, hi):
    return max(lo, min(hi, val))

def compute_flow_roi(cup_xyxy, img_w, img_h):
    """
    Flow ROI anchored so that its TOP aligns with the cup rim (cup box top).
    The ROI extends downward by roi_h. Keep LEFT edge fixed and trim RIGHT edge inward.
    """
    x1, y1, x2, y2 = cup_xyxy
    w = x2 - x1
    h = y2 - y1

    # Base ROI size relative to cup
    roi_h = int(max(MIN_ROI_H, ROI_HEIGHT_FRAC * h))
    roi_w = int(max(MIN_ROI_W, ROI_WIDTH_FRAC  * w))

    # Start centered, then trim right
    cx  = x1 + w / 2.0
    rx1 = int(cx - roi_w / 2.0)   # left edge (kept fixed)
    rx2 = rx1 + roi_w             # original right edge

    # Trim the RIGHT side only to avoid handle
    trim = int(RIGHT_INSET_FRAC * w)
    rx2  = rx2 - trim

    # Ensure ROI still has some width
    if rx2 - rx1 < MIN_ROI_W:
        rx1 = rx2 - MIN_ROI_W

    # Vertical placement: top at the rim, extend downward
    y_top    = int(y1)  # (optionally: int(y1 - RIM_GAP_FRAC*h))
    y_bottom = y_top + roi_h

    # Clip to image bounds
    rx1      = max(0, min(rx1, img_w - 1))
    rx2      = max(0, min(rx2, img_w - 1))
    y_top    = max(0, min(y_top,    img_h - 1))
    y_bottom = max(0, min(y_bottom, img_h - 1))

    if rx2 <= rx1 or y_bottom <= y_top:
        return None
    return (rx1, y_top, rx2, y_bottom)

def main():
    # Load YOLO
    model = YOLO(MODEL_WEIGHTS)
    names = getattr(model, "names", None) or getattr(model.model, "names", {})
    cup_ids = [i for i, n in (names.items() if isinstance(names, dict) else enumerate(names)) if str(n).lower() == "cup"]
    if not cup_ids:
        raise RuntimeError("Class 'cup' not found in model classes.")
    CUP_ID = int(cup_ids[0])

    # Collect frames
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, GLOB_PAT)))
    if not frames:
        print(f"No frames found under {INPUT_DIR}/{GLOB_PAT}")
        return

    # CSV
    csv_path = Path(OUTPUT_DIR) / "flow_rois.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "filename",
            "cup_x1","cup_y1","cup_x2","cup_y2","cup_conf",
            "roi_x1","roi_y1","roi_x2","roi_y2"
        ])

        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                print(f"Skip unreadable: {fp}")
                continue
            H, W = img.shape[:2]
            clean = img.copy()   # for analysis (no drawings)
            ann   = img.copy()   # for human-visible overlays

            # --- YOLO inference ---
            res = model.predict(source=img, verbose=False)[0]

            # Filter to cups above threshold; keep best by confidence
            best = None
            for b in res.boxes:  # each b has .cls, .conf, .xyxy
                cls_id = int(b.cls.item())
                conf   = float(b.conf.item())
                if cls_id == CUP_ID and conf >= CONF_THRESH:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    cand = (conf, (x1, y1, x2, y2))
                    if (best is None) or (cand[0] > best[0]):
                        best = cand

            if best is None:
                # Save CLEAN frame only, no drawings
                cv2.imwrite(os.path.join(CLEAN_DIR, Path(fp).name), clean)
                writer.writerow([Path(fp).name, "", "", "", "", "", "", "", ""])
                if WRITE_ANNOTATIONS:
                    cv2.imwrite(os.path.join(ANN_DIR, Path(fp).name), ann)
                continue

            # If we have a cup:
            cup_conf, cup_xyxy = best
            x1, y1, x2, y2 = cup_xyxy
            flow_roi = compute_flow_roi(cup_xyxy, W, H)

            # ---- SAVE CLEAN FRAME FIRST (no rectangles) ----
            cv2.imwrite(os.path.join(CLEAN_DIR, Path(fp).name), clean)

            # Optional: save an ROI mask so downstream can multiply/ignore non-ROI
            mask = np.zeros((H, W), np.uint8)
            if flow_roi is not None:
                rx1, ry1, rx2, ry2 = flow_roi
                cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)  # filled ROI
                writer.writerow([Path(fp).name, x1,y1,x2,y2,f"{cup_conf:.3f}", rx1,ry1,rx2,ry2])
            else:
                writer.writerow([Path(fp).name, x1,y1,x2,y2,f"{cup_conf:.3f}","","","",""])
            cv2.imwrite(os.path.join(MASK_DIR, Path(fp).stem + "_mask.png"), mask)

            # ---- ONLY draw on the annotated copy ----
            if WRITE_ANNOTATIONS:
                cv2.rectangle(ann, (x1,y1), (x2,y2), (0,255,0), 2)
                if flow_roi is not None:
                    rx1, ry1, rx2, ry2 = flow_roi
                    cv2.rectangle(ann, (rx1, ry1), (rx2, ry2), (0,0,255), 2)
                    cv2.putText(ann, f"flow-ROI (-{int(RIGHT_INSET_FRAC*100)}% right)",
                                (rx1, max(0, ry1-6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(ann, f"cup {cup_conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(ANN_DIR, Path(fp).name), ann)

    print(f"Clean frames → {CLEAN_DIR}")
    if WRITE_ANNOTATIONS:
        print(f"Annotated frames → {ANN_DIR}")
    print(f"ROI masks → {MASK_DIR}")
    print(f"CSV with ROIs → {csv_path}")
    print("process completed")

if __name__ == "__main__":
    main()
