import os
import csv
import re
from pathlib import Path
import cv2
import numpy as np
import easyocr

# ========= CONFIG =========
CSV_DETECTIONS = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_label_detection/all_objects_detection/detections_all_objects.csv"
CLEAN_DIR      = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_label_detection/all_objects_detection/clean_frames"
CROPS_DIR      = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_label_detection/all_objects_detection/bottle_crops_zoom25"
OCR_CSV_OUT    = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_label_detection/all_objects_detection/ocr_results_bottle_frames20_60.csv"

FRAME_MIN = 20
FRAME_MAX = 60
CLASS_NAME_TARGET = "bottle"
CLASS_ID_TARGETS = {39}  # COCO 'bottle' is 39; keep both name and id checks for safety.

# crop padding (relative to box size) to avoid cutting off label edges
PAD_RATIO = 0.10  # 10% each side
# zoom factor after cropping (resizing crop up)
ZOOM_FACTOR = 1.25

# OCR settings
LANGS = ['en']
USE_GPU = False  # set True if you want GPU and it's available

# ==========================
def frame_index_from_name(name):
    """expects names like frame_00051.jpg"""
    import re
    m = re.search(r"frame_(\d+)\.", name)
    return int(m.group(1)) if m else None


def pad_and_clip(x1, y1, x2, y2, W, H, pad_ratio=0.1):
    w = x2 - x1
    h = y2 - y1
    dx = int(round(w * pad_ratio))
    dy = int(round(h * pad_ratio))
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(W - 1, x2 + dx)
    ny2 = min(H - 1, y2 + dy)
    # ensure valid
    if nx2 <= nx1: nx2 = min(W - 1, nx1 + 1)
    if ny2 <= ny1: ny2 = min(H - 1, ny1 + 1)
    return nx1, ny1, nx2, ny2

def main():
    os.makedirs(CROPS_DIR, exist_ok=True)
    reader = easyocr.Reader(LANGS, gpu=USE_GPU)

    rows = []
    with open(CSV_DETECTIONS, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            fname = r["filename"].strip()
            if not fname:
                continue  # skip empty breadcrumb rows

            fidx = frame_index_from_name(fname)
            if fidx is None or fidx < FRAME_MIN or fidx > FRAME_MAX:
                continue

            # class checks (name or id)
            cls_name = (r.get("class_name") or "").strip().lower()
            cls_id_s = (r.get("class_id") or "").strip()
            try:
                cls_id = int(cls_id_s) if cls_id_s != "" else None
            except Exception:
                cls_id = None

            is_bottle = (cls_name == CLASS_NAME_TARGET) or (cls_id in CLASS_ID_TARGETS)
            if not is_bottle:
                continue

            # coords
            try:
                x1 = int(float(r["x1"])); y1 = int(float(r["y1"]))
                x2 = int(float(r["x2"])); y2 = int(float(r["y2"]))
            except Exception:
                continue

            rows.append((fname, x1, y1, x2, y2))

    if not rows:
        print("No matching bottle detections found in frames 20–60.")
        return

    ocr_out_rows = []
    for (fname, x1, y1, x2, y2) in rows:
        img_path = os.path.join(CLEAN_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue
        H, W = img.shape[:2]

        # pad and crop
        px1, py1, px2, py2 = pad_and_clip(x1, y1, x2, y2, W, H, PAD_RATIO)
        crop = img[py1:py2, px1:px2]

        # zoom (resize) by 1.25x (doesn't add new detail but can help OCR input size)
        zoom_w = max(1, int(round(crop.shape[1] * ZOOM_FACTOR)))
        zoom_h = max(1, int(round(crop.shape[0] * ZOOM_FACTOR)))
        crop_zoom = cv2.resize(crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

        # save crop
        base = Path(fname).stem
        crop_name = f"{base}_bottle_{px1}_{py1}_{px2}_{py2}_zoom125.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_name)
        cv2.imwrite(crop_path, crop_zoom)

        # OCR
        results = reader.readtext(crop_zoom, detail=False)  # list[str]
        text = " ".join(results).strip()
        print(f"{fname} → {text}")

        ocr_out_rows.append({
            "filename": fname,
            "crop_path": crop_path,
            "x1": px1, "y1": py1, "x2": px2, "y2": py2,
            "zoom": ZOOM_FACTOR,
            "text": text
        })

    # write OCR results CSV
    with open(OCR_CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename","crop_path","x1","y1","x2","y2","zoom","text"])
        w.writeheader()
        for r in ocr_out_rows:
            w.writerow(r)

    print("\n==== Done ====")
    print(f"Crops saved to: {CROPS_DIR}")
    print(f"OCR CSV saved to: {OCR_CSV_OUT}")

if __name__ == "__main__":
    main()
