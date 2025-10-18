#!/usr/bin/env python3
# unify_mouth_roi_labels_and_images.py
# - Uses nuanced labeling CSV to compute blender "mouth ROI" (VOI)
# - Writes augmented CSVs (ROI added to blender rows) under SecondPass/Metadata
# - Copies ALL images from nuancedlabeling to SecondPass/bounding_boxes
#   and overlays the mouth ROI only when a blender is present.
# - Originals remain untouched.

from pathlib import Path
import csv, re, cv2

# ======================= CONFIG =======================

# Inputs (read-only): nuanced labeling outputs
FUSED_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/nuancedlabeling")
IN_CSV    = FUSED_DIR / "all_detections_with_hand_side.csv"

# Second-pass outputs
OUT_META_ROOT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata")
OUT_META_DIR  = OUT_META_ROOT / "Metadata"
OUT_CSV       = OUT_META_DIR / "merged_labels_with_mouth_roi.csv"
ROI_ONLY_CSV  = OUT_META_DIR / "mouth_roi_only.csv"

OUT_IMG_ROOT = OUT_META_ROOT
OUT_IMG_DIR  = OUT_IMG_ROOT / "bounding_boxes"

# Filtering
TARGET_CLASS = "blender"
FRAME_MIN, FRAME_MAX = None, None  # leave None to include all frames

# Visualization
DRAW_MAIN_BOX   = False
BOX_COLOR       = (0, 255, 0)   # BGR for main blender box (optional)
MOUTH_COLOR     = (0, 0, 255)   # BGR for mouth ROI
THICK           = 2
LABEL_FONT      = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE     = 0.5
LABEL_THICK     = 1

# ROI geometry (as fractions of blender box)
MOUTH_BAND_FRAC  = 0.45  # vertical height of ROI as % of blender box height
LEFT_INSET_FRAC  = 0.10  # inset from left edge
RIGHT_INSET_FRAC = 0.10  # inset from right edge
RIM_GAP_FRAC     = 0.00  # start a bit below rim (0 = at rim)

# ======================= HELPERS ======================

def norm(s: str) -> str:
    return (s or "").strip().lower()

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
    rx1 = x1 + LEFT_INSET_FRAC  * w
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

# ======================= LOAD & AUGMENT ======================

def load_rows_and_blender_boxes():
    """Load all rows and collect blender boxes per frame for drawing."""
    with IN_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        in_rows = list(rdr)
        base_fields = rdr.fieldnames or []

    boxes_by_frame = {}
    for r in in_rows:
        if norm(r.get("cls_name", "")) != TARGET_CLASS:
            continue
        frame_name = r.get("frame") or r.get("source") or ""
        if not frame_name:
            continue
        idx = frame_index_from_name(frame_name)
        if FRAME_MIN is not None and (idx is None or idx < FRAME_MIN):
            continue
        if FRAME_MAX is not None and (idx is None or idx > FRAME_MAX):
            continue
        try:
            x1, y1, x2, y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
        except Exception:
            continue
        boxes_by_frame.setdefault(Path(frame_name).name, []).append((x1, y1, x2, y2))
    return in_rows, base_fields, boxes_by_frame

def write_augmented_csvs(in_rows, base_fields):
    """Write merged_labels_with_mouth_roi.csv and mouth_roi_only.csv (originals untouched)."""
    roi_fields = ["roi_name","roi_x1","roi_y1","roi_x2","roi_y2","roi_cx","roi_cy","roi_w","roi_h"]
    out_fields = list(base_fields)
    for c in roi_fields:
        if c not in out_fields:
            out_fields.append(c)

    with OUT_CSV.open("w", newline="") as f_out, ROI_ONLY_CSV.open("w", newline="") as f_roi:
        w_out = csv.DictWriter(f_out, fieldnames=out_fields)
        w_out.writeheader()

        w_roi = csv.writer(f_roi)
        w_roi.writerow([
            "frame","cls_name","obj_tid",
            "roi_name","roi_x1","roi_y1","roi_x2","roi_y2","roi_cx","roi_cy","roi_w","roi_h"
        ])

        updated, total = 0, 0
        for r in in_rows:
            total += 1
            r_out = dict(r)
            # initialize empty ROI fields (keeps schema consistent)
            for c in roi_fields:
                if c not in r_out:
                    r_out[c] = ""

            frame_name = r.get("frame") or r.get("source") or ""
            idx = frame_index_from_name(frame_name) if frame_name else None
            if FRAME_MIN is not None and (idx is None or idx < FRAME_MIN):
                w_out.writerow(r_out); continue
            if FRAME_MAX is not None and (idx is None or idx > FRAME_MAX):
                w_out.writerow(r_out); continue

            if norm(r.get("cls_name","")) == norm(TARGET_CLASS):
                try:
                    x1, y1, x2, y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
                except Exception:
                    w_out.writerow(r_out); continue

                roi = mouth_roi_from_box(x1, y1, x2, y2)
                if roi:
                    rx1, ry1, rx2, ry2 = roi
                    rcx = int(round((rx1 + rx2) / 2))
                    rcy = int(round((ry1 + ry2) / 2))
                    rw  = int(round(rx2 - rx1))
                    rh  = int(round(ry2 - ry1))

                    r_out["roi_name"] = "blender_mouth"
                    r_out["roi_x1"], r_out["roi_y1"] = str(rx1), str(ry1)
                    r_out["roi_x2"], r_out["roi_y2"] = str(rx2), str(ry2)
                    r_out["roi_cx"], r_out["roi_cy"] = str(rcx), str(rcy)
                    r_out["roi_w"],  r_out["roi_h"]  = str(rw),  str(rh)
                    updated += 1

                    w_roi.writerow([
                        frame_name, r.get("cls_name",""), r.get("obj_tid",""),
                        "blender_mouth", rx1, ry1, rx2, ry2, rcx, rcy, rw, rh
                    ])

            w_out.writerow(r_out)

    print(f"[META] Processed {total} rows; added ROI to {updated} blender rows.")
    print(f"[META] New merged labels → {OUT_CSV}")
    print(f"[META] ROI-only index → {ROI_ONLY_CSV}")

# ======================= IMAGES ======================

def draw_images_all_frames(boxes_by_frame):
    """
    For every image under FUSED_DIR:
      - If we have blender box(es), draw mouth ROI (and optional main box)
      - Otherwise, just copy the image unchanged
      - Always write to OUT_IMG_DIR
    """
    drawn, copied, missing = 0, 0, 0
    all_images = sorted(list(FUSED_DIR.glob("*.jpg")) + list(FUSED_DIR.glob("*.png")))

    for src in all_images:
        fname = src.name
        img = cv2.imread(str(src))
        if img is None:
            missing += 1
            continue

        # If blender present in this frame, draw ROI(s)
        if fname in boxes_by_frame:
            for (x1, y1, x2, y2) in boxes_by_frame[fname]:
                if DRAW_MAIN_BOX:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, THICK)

                roi = mouth_roi_from_box(x1, y1, x2, y2)
                if roi:
                    rx1, ry1, rx2, ry2 = roi
                    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), MOUTH_COLOR, THICK)
                    cv2.putText(img, "mouth-ROI", (rx1, max(0, ry1 - 6)),
                                LABEL_FONT, LABEL_SCALE, MOUTH_COLOR, LABEL_THICK, cv2.LINE_AA)
            drawn += 1
        else:
            copied += 1

        cv2.imwrite(str(OUT_IMG_DIR / fname), img)

    print(f"[IMG ] Annotated (with blender): {drawn}, copied (no blender): {copied} → {OUT_IMG_DIR}")
    if missing:
        print(f"[WARN] Missing/unreadable frames: {missing} from {FUSED_DIR}")

# ======================= MAIN ======================

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing nuanced labels: {IN_CSV}")
    ensure_dirs()
    in_rows, base_fields, boxes_by_frame = load_rows_and_blender_boxes()
    write_augmented_csvs(in_rows, base_fields)
    draw_images_all_frames(boxes_by_frame)
    print("[DONE] Second-pass metadata + full image set complete (originals unchanged).")

if __name__ == "__main__":
    main()
