#!/usr/bin/env python3
# compute_hand_to_mouth_distance.py
# Uses merged_labels_with_mouth_roi.csv to compute per-frame distance from
# blender mouth ROI centroid to the nearest hand (or person) detection.

from pathlib import Path
import csv
import re
import math
import cv2

# =======================
# ======= CONFIG ========
# =======================

# Inputs
FUSED_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/FusedPreview")
IN_CSV    = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/merged_labels_with_mouth_roi.csv")

# Outputs (metadata + optional images)
OUT_META_ROOT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata")
OUT_META_DIR  = OUT_META_ROOT / "Metadata"
OUT_CSV       = OUT_META_DIR / "hand_to_mouth_distance.csv"

OUT_IMG_ROOT  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata")
OUT_IMG_DIR   = OUT_IMG_ROOT / "HandDistance"   # annotated previews (optional)

# Frame filtering (None = all)
FRAME_MIN, FRAME_MAX = None, None   # e.g., 0,100 to limit

# What to consider as "hand"
# - If HAND_ONLY=True, restricts to classes that start with "hand" (e.g., hand_open, hand_closed, hand_pinch)
# - If False, also allows "person" boxes as a fallback (useful if hand classes are sparse)
HAND_ONLY = False
HAND_PREFIXES = ("hand",)          # any cls_name that startswith one of these
ALLOW_PERSON  = True               # only used when HAND_ONLY=False

# Visualization
WRITE_ANNOTATIONS   = True          # set False to skip images
DRAW_LIMIT          = 800          # annotate at most this many frames (None = no cap)
CLR_ROI             = (0, 128, 255) # orange
CLR_HAND            = (0,   0, 255) # red
THICK               = 2
RADIUS_PT           = 5
FONT                = cv2.FONT_HERSHEY_SIMPLEX

# =======================
# ======= LOGIC =========
# =======================

def norm(s: str) -> str:
    return (s or "").strip().lower()

def is_handlike(cls: str) -> bool:
    c = norm(cls)
    if any(c.startswith(p) for p in HAND_PREFIXES):
        return True
    if (not HAND_ONLY) and ALLOW_PERSON and c == "person":
        return True
    return False

def frame_index_from_name(name: str):
    m = re.search(r"(\d+)$", Path(name).stem)
    return int(m.group(1)) if m else None

def diag(w, h):
    try:
        return math.hypot(float(w), float(h))
    except:
        return None

def ensure_dirs():
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
    if WRITE_ANNOTATIONS:
        OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

def load_rows():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing: {IN_CSV}")
    with IN_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        fields = rdr.fieldnames or []
    return rows, fields

def collect_per_frame(rows):
    """
    Build:
      roi_by_frame[frame] = (roi_cx, roi_cy, img_w, img_h, (rx1,ry1,rx2,ry2))
      hands_by_frame[frame] = list of dicts {tid, cx, cy, cls_name, (x1,y1,x2,y2)}
    """
    roi_by_frame = {}
    hands_by_frame = {}

    for r in rows:
        frame = r.get("frame") or r.get("source") or ""
        if not frame:
            continue
        idx = frame_index_from_name(frame)
        if FRAME_MIN is not None and (idx is None or idx < FRAME_MIN): 
            continue
        if FRAME_MAX is not None and (idx is None or idx > FRAME_MAX): 
            continue

        img_w = r.get("img_w") or r.get("width") or ""
        img_h = r.get("img_h") or r.get("height") or ""
        try:
            img_w = int(float(img_w))
            img_h = int(float(img_h))
        except:
            img_w = img_w or ""
            img_h = img_h or ""

        # Capture ROI if this row has one (only need one per frame; use first found)
        roi_name = r.get("roi_name", "")
        if roi_name:
            try:
                rx1 = int(float(r.get("roi_x1",""))); ry1 = int(float(r.get("roi_y1","")))
                rx2 = int(float(r.get("roi_x2",""))); ry2 = int(float(r.get("roi_y2","")))
                rcx = int(float(r.get("roi_cx","")));  rcy = int(float(r.get("roi_cy","")))
                roi_by_frame.setdefault(Path(frame).name, (rcx, rcy, img_w, img_h, (rx1,ry1,rx2,ry2)))
            except:
                pass  # ignore malformed ROI

        # Collect hand-like detections
        cls = r.get("cls_name", "")
        if is_handlike(cls):
            try:
                x1 = float(r["x1"]); y1 = float(r["y1"])
                x2 = float(r["x2"]); y2 = float(r["y2"])
                cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5
                tid = r.get("obj_tid","")
                hands_by_frame.setdefault(Path(frame).name, []).append({
                    "tid": tid, "cx": cx, "cy": cy, "cls_name": cls,
                    "box": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                })
            except:
                pass

    return roi_by_frame, hands_by_frame

def nearest_hand(roi_cx, roi_cy, hand_list):
    if not hand_list:
        return None, None
    best = None
    best_d = 1e18
    for h in hand_list:
        d = math.hypot(h["cx"] - roi_cx, h["cy"] - roi_cy)
        if d < best_d:
            best_d = d
            best = h
    return best, best_d

def annotate_and_save(frame_name, roi, hand, img_cap_count):
    """Draw ROI center, nearest hand center and a line; save preview image."""
    if not WRITE_ANNOTATIONS:
        return img_cap_count
    src = FUSED_DIR / frame_name
    if not src.exists():
        return img_cap_count
    if (DRAW_LIMIT is not None) and (img_cap_count >= DRAW_LIMIT):
        return img_cap_count

    img = cv2.imread(str(src))
    if img is None:
        return img_cap_count
    H, W = img.shape[:2]

    rcx, rcy, _, _, (rx1,ry1,rx2,ry2) = roi
    # ROI rectangle
    cv2.rectangle(img, (rx1,ry1), (rx2,ry2), CLR_ROI, THICK)
    cv2.circle(img, (int(rcx), int(rcy)), RADIUS_PT, CLR_ROI, -1)
    cv2.putText(img, "mouth-ROI", (rx1, max(0, ry1-6)), FONT, 0.5, CLR_ROI, 1, cv2.LINE_AA)

    if hand is not None:
        hx, hy = int(round(hand["cx"])), int(round(hand["cy"]))
        cv2.circle(img, (hx, hy), RADIUS_PT, CLR_HAND, -1)
        cv2.line(img, (int(rcx), int(rcy)), (hx, hy), CLR_HAND, THICK)
        cv2.putText(img, norm(hand["cls_name"]), (hx+6, max(0, hy-6)), FONT, 0.5, CLR_HAND, 1, cv2.LINE_AA)

    dst = OUT_IMG_DIR / frame_name
    cv2.imwrite(str(dst), img)
    return img_cap_count + 1

def main():
    ensure_dirs()
    rows, _ = load_rows()
    roi_by_frame, hands_by_frame = collect_per_frame(rows)

    # Write per-frame nearest-hand distances
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame","img_w","img_h",
            "roi_cx","roi_cy","roi_x1","roi_y1","roi_x2","roi_y2",
            "nearest_hand_tid","nearest_hand_cls","hand_cx","hand_cy",
            "dist_px","dist_fracdiag"
        ])

        img_cap = 0
        for frame_name, roi in sorted(roi_by_frame.items()):
            rcx, rcy, img_w, img_h, (rx1,ry1,rx2,ry2) = roi
            hand_list = hands_by_frame.get(frame_name, [])
            hbest, d_px = nearest_hand(rcx, rcy, hand_list)
            d_frac = ""
            if (img_w and img_h) and (d_px is not None):
                D = diag(img_w, img_h) or 1.0
                d_frac = d_px / D

            w.writerow([
                frame_name, img_w, img_h,
                rcx, rcy, rx1, ry1, rx2, ry2,
                (hbest["tid"] if hbest else ""),
                (hbest["cls_name"] if hbest else ""),
                (f"{hbest['cx']:.1f}" if hbest else ""),
                (f"{hbest['cy']:.1f}" if hbest else ""),
                (f"{d_px:.1f}" if d_px is not None else ""),
                (f"{d_frac:.4f}" if d_frac != "" else "")
            ])

            # preview image
            img_cap = annotate_and_save(frame_name, roi, hbest, img_cap)

    print(f"[DONE] Wrote distances → {OUT_CSV}")
    if WRITE_ANNOTATIONS:
        print(f"[DONE] Annotated previews → {OUT_IMG_DIR}")

if __name__ == "__main__":
    main()
