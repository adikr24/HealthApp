## Detect ALL objects with YOLO and save annotated frames + CSV ##

import os
import glob
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ======= Paths (edit if needed) =======
INPUT_DIR   = "/app/mediaFiles/videos/AnnotateVideos/RawVideos/milkPour"
OUTPUT_DIR  = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_label_detection/all_objects_detection"
GLOB_PAT    = "frame_*.jpg"         # your frame naming pattern
MODEL_WEIGHTS = "/app/mediaFiles/yolo_results/YoloWeights/yolov8n.pt"
CONF_THRESH = 0.35                   # YOLO confidence threshold for ALL classes
LINE_THICK  = 2                      # rectangle thickness
FONT_SCALE  = 0.5
FONT_THICK  = 1

# Save clean vs. annotated outputs separately
WRITE_ANNOTATIONS = True
ANN_DIR   = os.path.join(OUTPUT_DIR, "annotated_frames")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

def color_for_class(cls_id: int):
    """Deterministic BGR color per class id."""
    # simple palette hashed by class id
    np.random.seed(cls_id + 7)
    c = tuple(int(x) for x in np.random.randint(0, 255, size=3))
    return (c[0], c[1], c[2])  # already BGR for OpenCV

def draw_label(img, text, x1, y1, color=(0,255,0)):
    """Draw text with filled background above the box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    pad = 2
    y1 = max(th + 2*pad, y1)
    cv2.rectangle(img, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK, cv2.LINE_AA)

def main():
    # Load YOLO
    model = YOLO(MODEL_WEIGHTS)

    # Resolve class-name mapping robustly (list or dict)
    names_attr = getattr(model, "names", None) or getattr(model.model, "names", None)
    if isinstance(names_attr, dict):
        id_to_name = dict(names_attr)
    else:
        id_to_name = {i: n for i, n in enumerate(names_attr)} if names_attr is not None else {}

    # Collect frames
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, GLOB_PAT)))
    if not frames:
        print(f"No frames found under {INPUT_DIR}/{GLOB_PAT}")
        return

    # CSV path
    csv_path = Path(OUTPUT_DIR) / "detections_all_objects.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "filename", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2"
        ])

        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                print(f"Skip unreadable: {fp}")
                continue
            H, W = img.shape[:2]
            clean = img.copy()
            ann   = img.copy()

            # --- YOLO inference on this image ---
            # Setting conf here reduces post-filtering work
            res = model.predict(source=img, conf=CONF_THRESH, verbose=False)[0]

            det_count = 0
            if res.boxes is not None and len(res.boxes) > 0:
                # Iterate all detections above threshold
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    conf   = float(b.conf.item())
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                    # Clip to image bounds (safety)
                    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cls_name = str(id_to_name.get(cls_id, cls_id))
                    writer.writerow([Path(fp).name, cls_id, cls_name, f"{conf:.3f}", x1, y1, x2, y2])
                    det_count += 1

                    # Draw on annotated copy
                    if WRITE_ANNOTATIONS:
                        color = color_for_class(cls_id)
                        cv2.rectangle(ann, (x1, y1), (x2, y2), color, LINE_THICK)
                        draw_label(ann, f"{cls_name} {conf:.2f}", x1, max(0, y1 - 4), color)

            # If no detections, still leave a breadcrumb in CSV (optional)
            if det_count == 0:
                writer.writerow([Path(fp).name, "", "", "", "", "", "", ""])

            # Save frames
            cv2.imwrite(os.path.join(CLEAN_DIR, Path(fp).name), clean)
            if WRITE_ANNOTATIONS:
                cv2.imwrite(os.path.join(ANN_DIR, Path(fp).name), ann)

    print(f"Clean frames  → {CLEAN_DIR}")
    if WRITE_ANNOTATIONS:
        print(f"Annotated → {ANN_DIR}")
    print(f"CSV (all detections) → {csv_path}")
    print("process completed")

if __name__ == "__main__":
    main()
