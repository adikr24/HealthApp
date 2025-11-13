#!/usr/bin/env python3
# add_hand_rois_inside_v3.py
# v3 = v2 + merge in NON-HAND detections so the output CSV is a single source of truth.

from pathlib import Path
import csv, cv2, re, math
from collections import Counter, defaultdict

# ======================= PATHS =======================

BASE_SECOND = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata")

# Primary hand-labeled inputs (first existing is used)
PRIMARY_LABELS = [
    BASE_SECOND / "Metadata" / "HandActivity" / "hand_label_collapse.csv",  # preferred (hands-only or hands-merged)
    BASE_SECOND / "Metadata" / "voi_bbox_and_hands_labels.csv",             # hands + VOI + maybe some objs
    BASE_SECOND / "Metadata" / "all_detections_with_hand_side.csv",                       # fallback (may contain hands+objs)
]

# Supplemental YOLO objects to merge (non-hand)
SUPP_OBJECTS = [
    BASE_SECOND / "Metadata" / "all_detections_with_hand_side.csv",  # preferred location in 2nd pass
    BASE_SECOND / "Metadata" / "all_detections_with_hand_side.csv",             # fallback
]

# Images
VOI_IMG_DIR = BASE_SECOND / "bounding_boxes" / "VOIBoundingBoxes"
SRC_IMG_DIR = BASE_FIRST

# Outputs
OUT_META_DIR = BASE_SECOND / "Metadata" / "HandActivity"
OUT_IMG_DIR  = BASE_SECOND / "bounding_boxes" / "HandROIs"
OUT_CSV      = OUT_META_DIR / "hand_rois.csv"
OUT_ERR_CSV  = OUT_META_DIR / "hand_roi_errors.csv"

OUT_META_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ======================= CONFIG =======================

HAND_COMPONENT_LABELS = {"left_hand", "right_hand", "unknown_hand", "merged_hands"}

# ROI sizing (inside hand box)
W_FRAC = 0.30
H_FRAC = 0.30
SIDE_INSET_FRAC = 0.02
TOP_INSET_FRAC  = 0.02
MIN_PX = 6

# For merged_hands (top-center band) trims
MERGED_TRIM_LEFT_FRAC  = 0.15
MERGED_TRIM_RIGHT_FRAC = 0.15

# Drawing
HAND_ROI_COLOR = (255, 200, 0)   # BGR
HAND_BOX_COLOR = (60, 200, 255)
THICK          = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
TSCALE = 0.5
TTHICK = 1

# ======================= HELPERS =======================

def norm(s): 
    return (s or "").strip().lower()

def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            print(f"[INFO] Using: {p}")
            return p
    return None

def frame_idx(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else -1

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_f(v, default=None):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"", "none", "nan", "inf", "-inf"}:
        return default
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default

def valid_box(x1,y1,x2,y2):
    return (x1 is not None and y1 is not None and x2 is not None and y2 is not None
            and (x2 > x1) and (y2 > y1))

def is_handish(label_or_cls: str) -> bool:
    n = norm(label_or_cls)
    return ("hand" in n) or ("palm" in n) or ("finger" in n)

def row_to_hand_label_and_box(row):
    """
    Normalize to (label, box) for hand components.
    Priority:
      1) 'label' column: left_hand/right_hand/merged_hands/unknown_hand
      2) cls_name_* already a hand component
      3) raw 'hand' + hand_side -> left_hand/right_hand
    """
    # 1) collapse format
    lbl = norm(row.get("label"))
    if lbl in HAND_COMPONENT_LABELS:
        x1 = to_f(row.get("x1")); y1 = to_f(row.get("y1"))
        x2 = to_f(row.get("x2")); y2 = to_f(row.get("y2"))
        if valid_box(x1,y1,x2,y2):
            return lbl, (x1,y1,x2,y2)
        return None, None

    # 2) unified/raw with class names already as components
    cls = norm(row.get("cls_name_norm") or row.get("cls_name") or row.get("cls_name_orig"))
    if cls in HAND_COMPONENT_LABELS:
        x1 = to_f(row.get("x1")); y1 = to_f(row.get("y1"))
        x2 = to_f(row.get("x2")); y2 = to_f(row.get("y2"))
        if valid_box(x1,y1,x2,y2):
            return cls, (x1,y1,x2,y2)
        return None, None

    # 3) raw: "hand" + hand_side
    if cls == "hand":
        side = norm(row.get("hand_side"))
        if side in ("left","right"):
            lbl = f"{side}_hand"
            x1 = to_f(row.get("x1")); y1 = to_f(row.get("y1"))
            x2 = to_f(row.get("x2")); y2 = to_f(row.get("y2"))
            if valid_box(x1,y1,x2,y2):
                return lbl, (x1,y1,x2,y2)

    return None, None

def hand_roi_inside(label: str, box):
    """
    Returns (rx1, ry1, rx2, ry2) ints or None if invalid/tiny.
    """
    x1,y1,x2,y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w < 2*MIN_PX or h < 2*MIN_PX:
        return None

    rw = max(MIN_PX, W_FRAC * w)
    rh = max(MIN_PX, H_FRAC * h)
    padL = SIDE_INSET_FRAC * w
    padR = SIDE_INSET_FRAC * w
    padT = TOP_INSET_FRAC  * h

    if label == "left_hand":      # top-right
        rx2 = x2 - padR;    rx1 = rx2 - rw
        ry1 = y1 + padT;    ry2 = ry1 + rh
    elif label == "right_hand":   # top-left
        rx1 = x1 + padL;    rx2 = rx1 + rw
        ry1 = y1 + padT;    ry2 = ry1 + rh
    elif label == "merged_hands": # top-center band with trims
        rx1 = x1 + MERGED_TRIM_LEFT_FRAC  * w
        rx2 = x2 - MERGED_TRIM_RIGHT_FRAC * w
        if (rx2 - rx1) < rw:
            cx = (x1 + x2) / 2.0
            rx1, rx2 = cx - rw/2.0, cx + rw/2.0
        ry1 = y1 + padT;    ry2 = ry1 + rh
    else:                         # unknown_hand → top-center
        cx = (x1 + x2) / 2.0
        rx1, rx2 = cx - rw/2.0, cx + rw/2.0
        ry1 = y1 + padT;    ry2 = ry1 + rh

    rx1 = clamp(rx1, x1, x2);  rx2 = clamp(rx2, x1, x2)
    ry1 = clamp(ry1, y1, y2);  ry2 = clamp(ry2, y1, y2)

    if (rx2 - rx1) < MIN_PX:
        rx2 = clamp(rx1 + MIN_PX, x1, x2)
        rx1 = clamp(rx2 - MIN_PX, x1, x2)
    if (ry2 - ry1) < MIN_PX:
        ry2 = clamp(ry1 + MIN_PX, y1, y2)
        ry1 = clamp(ry2 - MIN_PX, y1, y2)

    if rx2 <= rx1 or ry2 <= ry1:
        return None

    return tuple(int(round(v)) for v in (rx1, ry1, rx2, ry2))

def dedupe_key(r):
    """A rough dedupe key for merging: frame + class + rounded box."""
    frame = Path(r.get("frame") or "").name
    cls = norm(r.get("cls_name_norm") or r.get("cls_name") or r.get("cls_name_orig") or r.get("label"))
    x1 = to_f(r.get("x1")); y1 = to_f(r.get("y1"))
    x2 = to_f(r.get("x2")); y2 = to_f(r.get("y2"))
    if None in (x1,y1,x2,y2):
        return None
    return (frame, cls, round(x1,1), round(y1,1), round(x2,1), round(y2,1))

# ======================= MAIN =======================

def main():
    # ---- Pick primary hands file
    primary_csv = pick_first_existing(PRIMARY_LABELS)
    if primary_csv is None:
        raise FileNotFoundError(f"No input labels found. Checked: {PRIMARY_LABELS}")

    # ---- Load primary rows
    with primary_csv.open("r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        primary_rows = list(rdr)
        fields = list(rdr.fieldnames or [])

    # ---- Optionally load supplemental objects
    supp_csv = pick_first_existing(SUPP_OBJECTS)
    supp_rows = []
    if supp_csv and supp_csv != primary_csv:
        with supp_csv.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            supp_rows = list(rdr)

    # ---- Stats
    def seen_label(r):
        return norm(r.get("label") or r.get("cls_name_norm") or r.get("cls_name") or r.get("cls_name_orig"))
    lbl_counts = Counter(seen_label(r) for r in primary_rows)
    print("[DEBUG] Primary label/class counts:", dict(lbl_counts))
    if supp_rows:
        print(f"[DEBUG] Loaded supplemental objects: {len(supp_rows)} rows from {supp_csv.name}")

    # ---- Output header: original + hand_roi_* (even for non-hand rows)
    extra_cols = ["hand_roi_x1","hand_roi_y1","hand_roi_x2","hand_roi_y2",
                  "hand_roi_cx","hand_roi_cy","hand_roi_w","hand_roi_h"]
    out_fields = list(fields)
    for c in extra_cols:
        if c not in out_fields:
            out_fields.append(c)
    # ensure common detection fields exist
    for c in ["cls_id","cls_name_orig","cls_name_norm","conf","hand_side"]:
        if c not in out_fields:
            out_fields.append(c)

    # ---- Group by frame for image drawing
    by_frame = defaultdict(list)
    for r in primary_rows:
        fname = Path(r.get("frame") or "").name
        if fname:
            by_frame[fname].append(r)

    # ---- Build a set of keys to avoid duplicates when merging
    existing_keys = set()
    for r in primary_rows:
        k = dedupe_key(r)
        if k: existing_keys.add(k)

    # ---- Merge supplemental non-hand detections
    merged_new = 0
    for r in supp_rows:
        # skip hands; we only want *non-hand* here
        cls = norm(r.get("cls_name_norm") or r.get("cls_name") or r.get("cls_name_orig"))
        lbl = norm(r.get("label"))
        looks_hand = is_handish(cls) or is_handish(lbl)
        if looks_hand:
            continue
        k = dedupe_key(r)
        if not k or k in existing_keys:
            continue
        fname = Path(r.get("frame") or "").name
        if not fname:
            continue
        by_frame[fname].append(r)
        existing_keys.add(k)
        merged_new += 1
    if merged_new:
        print(f"[INFO] Merged in {merged_new} non-hand rows from supplemental detections.")

    # ---- Choose image source
    use_voi_imgs = VOI_IMG_DIR.exists()
    src_img_dir = VOI_IMG_DIR if use_voi_imgs else SRC_IMG_DIR

    out_rows   = []
    err_rows   = []
    err_counts = Counter()
    hands_total = 0
    hands_with_roi = 0

    for fname in sorted(by_frame, key=frame_idx):
        img_path = src_img_dir / fname
        img = cv2.imread(str(img_path)) if img_path.exists() else None

        for r in by_frame[fname]:
            out = dict(r)
            # normalize common fields presence
            for c in ["cls_id","cls_name_orig","cls_name_norm","conf","hand_side"]:
                out.setdefault(c, r.get(c, ""))
            # add ROI columns
            for c in extra_cols:
                out.setdefault(c, "")

            # If this row represents a hand → compute ROI
            lbl, box = row_to_hand_label_and_box(r)
            if lbl and box:
                hands_total += 1
                hroi = hand_roi_inside(lbl, box)
                if hroi:
                    rx1, ry1, rx2, ry2 = hroi
                    rcx = (rx1 + rx2) // 2
                    rcy = (ry1 + ry2) // 2
                    rw  = (rx2 - rx1)
                    rh  = (ry2 - ry1)
                    out.update({
                        "hand_roi_x1": rx1, "hand_roi_y1": ry1,
                        "hand_roi_x2": rx2, "hand_roi_y2": ry2,
                        "hand_roi_cx": rcx, "hand_roi_cy": rcy,
                        "hand_roi_w": rw,   "hand_roi_h": rh
                    })
                    # for downstream convenience, ensure label column carries the resolved hand label
                    out["label"] = lbl
                    hands_with_roi += 1

                    if img is not None:
                        try:
                            bx1 = int(round(to_f(r.get("x1")))); by1 = int(round(to_f(r.get("y1"))))
                            bx2 = int(round(to_f(r.get("x2")))); by2 = int(round(to_f(r.get("y2"))))
                            cv2.rectangle(img, (bx1,by1), (bx2,by2), HAND_BOX_COLOR, 1)
                            cv2.putText(img, lbl, (bx1, max(10,by1-6)), FONT, TSCALE, HAND_BOX_COLOR, 1, cv2.LINE_AA)
                        except Exception:
                            pass
                        cv2.rectangle(img, (rx1,ry1), (rx2,ry2), HAND_ROI_COLOR, THICK)
                        cv2.putText(img, "hand-ROI", (rx1, max(10, ry1-6)), FONT, TSCALE, HAND_ROI_COLOR, TTHICK, cv2.LINE_AA)
                else:
                    # Could not build ROI → record reason
                    x1 = to_f(r.get("x1")); y1 = to_f(r.get("y1"))
                    x2 = to_f(r.get("x2")); y2 = to_f(r.get("y2"))
                    reason = "roi_construction_failed"
                    if not valid_box(x1,y1,x2,y2):
                        reason = "invalid_hand_bbox"
                    else:
                        w = (x2 - x1) if (x2 is not None and x1 is not None) else None
                        h = (y2 - y1) if (y2 is not None and y1 is not None) else None
                        if (w is not None and h is not None) and (w < 2*MIN_PX or h < 2*MIN_PX):
                            reason = "tiny_hand_bbox"
                    err_rows.append({
                        "frame": fname,
                        "hand_label": lbl,
                        "reason": reason,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })
                    err_counts[reason] += 1
            else:
                # Non-hand row → pass through unmodified (ROI fields remain blank)
                pass

            out_rows.append(out)

        if img is not None:
            OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT_IMG_DIR / fname), img)

    # ---- Write unified CSV (hands + non-hands)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    # ---- Write ROI errors (even if empty)
    with OUT_ERR_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame","hand_label","reason","x1","y1","x2","y2"])
        w.writeheader()
        w.writerows(err_rows)

    # ---- Summary
    print(f"[DONE] Unified CSV (hands + non-hands) → {OUT_CSV}  (rows={len(out_rows)})")
    print(f"[DONE] Annotated frames → {OUT_IMG_DIR}")
    print(f"[INFO] Hands total: {hands_total} | with ROI: {hands_with_roi} | ROI failures: {len(err_rows)}")
    if err_counts:
        print("[INFO] Failures by reason:", dict(err_counts))
    print(f"[INFO] ROI error audit → {OUT_ERR_CSV}")

if __name__ == "__main__":
    main()
