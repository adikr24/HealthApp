#!/usr/bin/env python3
# unify_mouth_roi_labels_and_images.py
# - Reads nuanced YOLO CSV (hands + objects)
# - Finds blenders and computes their "mouth ROI" (VOI)
# - Writes voi_bbox_and_hands_labels.csv under SecondPass/Metadata
# - Copies all frames to SecondPass/bounding_boxes/VOIBoundingBoxes
#   and overlays red ROI box on blender mouth region
# - Originals remain untouched.

from pathlib import Path
import csv, re, cv2, shutil
from collections import Counter
from pathlib import Path as _P

# ======================= CONFIG =======================

# Input (read-only)
FUSED_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/detailed_hand_sides"
)
IN_CSV = FUSED_DIR / "all_detections_with_hand_side.csv"

# Second-pass outputs
OUT_META_ROOT = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata"
)
OUT_META_DIR = OUT_META_ROOT / "Metadata"
OUT_IMG_DIR = OUT_META_ROOT / "bounding_boxes" / "VOIBoundingBoxes"  # <- organized
OUT_CSV = OUT_META_DIR / "voi_bbox_and_hands_labels.csv"

# Copy these source CSVs from FUSED_DIR -> OUT_META_DIR if present
SOURCE_CSV_NAMES = [
    "all_detections_with_hand_side.csv",
    "labels.csv",  # optional
]

# Class filtering
TARGET_CLASS = "blender"
FRAME_MIN, FRAME_MAX = None, None  # None = include all frames

# Visualization
DRAW_MAIN_BOX = False
BOX_COLOR = (0, 255, 0)     # green for main blender box
MOUTH_COLOR = (0, 0, 255)   # red for ROI
THICK = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
LABEL_THICK = 1

# ROI geometry (as fractions of blender box)
MOUTH_BAND_FRAC = 0.45   # vertical height of ROI as % of blender box height
LEFT_INSET_FRAC = 0.10   # inset from left edge
RIGHT_INSET_FRAC = 0.10  # inset from right edge
RIM_GAP_FRAC = 0.00      # start a bit below rim (0 = at rim)

# =======================================================

def norm(s: str) -> str:
    return (s or "").strip().lower()

def get_cls(row: dict) -> str:
    """Return normalized class name (handles cls_name_norm, cls_name, cls_name_orig)."""
    return norm(
        row.get("cls_name_norm") or row.get("cls_name") or row.get("cls_name_orig") or ""
    )

def frame_index_from_name(name: str):
    """Extract trailing integer from filenames like 'frame_00063.jpg'."""
    m = re.search(r"(\d+)$", Path(name).stem)
    return int(m.group(1)) if m else None

def mouth_roi_from_box(x1, y1, x2, y2):
    """Return (rx1,ry1,rx2,ry2) ints for a top-band ROI inside the blender box."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    rx1 = x1 + LEFT_INSET_FRAC * w
    rx2 = x2 - RIGHT_INSET_FRAC * w
    band_h = max(2.0, MOUTH_BAND_FRAC * h)
    ry1 = y1 + RIM_GAP_FRAC * h
    ry2 = ry1 + band_h
    rx1, ry1, rx2, ry2 = int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return (rx1, ry1, rx2, ry2)

def ensure_dirs():
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def copy_source_csvs():
    copied = 0
    for name in SOURCE_CSV_NAMES:
        src = FUSED_DIR / name
        if src.exists():
            shutil.copy2(src, OUT_META_DIR / name)
            copied += 1
    if copied:
        print(f"[COPY] Copied {copied} CSV(s) from {FUSED_DIR.name} → {OUT_META_DIR}")

# =======================================================
# LOAD + AUGMENT
# =======================================================

def load_rows_and_blender_boxes():
    """Load all rows and collect blender boxes per frame for drawing."""
    with IN_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        in_rows = list(rdr)
        base_fields = rdr.fieldnames or []

    print("[DEBUG] Class counts:", dict(Counter(get_cls(r) for r in in_rows)))

    boxes_by_frame = {}
    for r in in_rows:
        if get_cls(r) != TARGET_CLASS:
            continue
        frame_name = r.get("frame") or r.get("source") or ""
        if not frame_name:
            continue
        idx = frame_index_from_name(frame_name)
        if FRAME_MIN and (idx is None or idx < FRAME_MIN):
            continue
        if FRAME_MAX and (idx is None or idx > FRAME_MAX):
            continue
        try:
            x1, y1, x2, y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
        except Exception:
            continue
        boxes_by_frame.setdefault(Path(frame_name).name, []).append((x1, y1, x2, y2))
    return in_rows, base_fields, boxes_by_frame

def write_augmented_csv(in_rows, base_fields, flow_w):
    """Write voi_bbox_and_hands_labels.csv (ROI added to blender rows)."""
    roi_fields = [
        "roi_name", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "roi_cx", "roi_cy", "roi_w", "roi_h"
    ]
    out_fields = list(base_fields)
    for c in roi_fields:
        if c not in out_fields:
            out_fields.append(c)

    total, updated = 0, 0
    with OUT_CSV.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        writer.writeheader()

        for r in in_rows:
            total += 1
            r_out = dict(r)
            for c in roi_fields:
                r_out.setdefault(c, "")

            frame_name = r.get("frame") or r.get("source") or ""
            idx = frame_index_from_name(frame_name) if frame_name else None
            if FRAME_MIN and (idx is None or idx < FRAME_MIN):
                writer.writerow(r_out)
                continue
            if FRAME_MAX and (idx is None or idx > FRAME_MAX):
                writer.writerow(r_out)
                continue

            if get_cls(r) == TARGET_CLASS:
                try:
                    x1, y1, x2, y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
                except Exception:
                    writer.writerow(r_out)
                    continue

                roi = mouth_roi_from_box(x1, y1, x2, y2)
                if roi:
                    rx1, ry1, rx2, ry2 = roi
                    rcx = int(round((rx1 + rx2) / 2))
                    rcy = int(round((ry1 + ry2) / 2))
                    rw = int(round(rx2 - rx1))
                    rh = int(round(ry2 - ry1))

                    r_out.update({
                        "roi_name": "blender",
                        "roi_x1": rx1, "roi_y1": ry1,
                        "roi_x2": rx2, "roi_y2": ry2,
                        "roi_cx": rcx, "roi_cy": rcy,
                        "roi_w": rw, "roi_h": rh
                    })
                    updated += 1

                    # === NEW: write a row to flow_rois.csv ===
                    
                    fname = _P(frame_name).name if frame_name else ""
                    flow_w.writerow([fname, rx1, ry1, rx2, ry2])


            writer.writerow(r_out)

    print(f"[META] Processed {total} rows; added ROI to {updated} blender rows.")
    print(f"[META] New merged labels → {OUT_CSV}")

# =======================================================
# IMAGE DRAWING
# =======================================================

def draw_images_all_frames(boxes_by_frame):
    drawn, copied, missing = 0, 0, 0
    all_images = sorted(list(FUSED_DIR.glob("*.jpg")) + list(FUSED_DIR.glob("*.png")))

    for src in all_images:
        fname = src.name
        img = cv2.imread(str(src))
        if img is None:
            missing += 1
            continue

        if fname in boxes_by_frame:
            for (x1, y1, x2, y2) in boxes_by_frame[fname]:
                if DRAW_MAIN_BOX:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, THICK)
                roi = mouth_roi_from_box(x1, y1, x2, y2)
                if roi:
                    rx1, ry1, rx2, ry2 = roi
                    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), MOUTH_COLOR, THICK)
                    cv2.putText(
                        img, "mouth-ROI", (rx1, max(0, ry1 - 6)),
                        LABEL_FONT, LABEL_SCALE, MOUTH_COLOR, LABEL_THICK, cv2.LINE_AA
                    )
            drawn += 1
        else:
            copied += 1

        cv2.imwrite(str(OUT_IMG_DIR / fname), img)

    print(f"[IMG ] Annotated (with blender): {drawn}, copied (no blender): {copied} → {OUT_IMG_DIR}")
    if missing:
        print(f"[WARN] Missing/unreadable frames: {missing} from {FUSED_DIR}")

# =======================================================
# MAIN
# =======================================================

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing nuanced labels: {IN_CSV}")
    ensure_dirs()
    # prepare flow_rois.csv
    flow_rois_path = OUT_META_DIR / "flow_rois.csv"
    flow_f = flow_rois_path.open("w", newline="")
    flow_w = csv.writer(flow_f)
    flow_w.writerow(["filename","roi_x1","roi_y1","roi_x2","roi_y2"])

    copy_source_csvs()
    in_rows, base_fields, boxes_by_frame = load_rows_and_blender_boxes()

    # pass writer here
    write_augmented_csv(in_rows, base_fields, flow_w)
    flow_f.close()
    print(f"[META] flow_rois.csv → {flow_rois_path}")

    draw_images_all_frames(boxes_by_frame)
    print("[DONE] VOI metadata + ROI overlays complete (originals untouched).")

if __name__ == "__main__":
    main()
