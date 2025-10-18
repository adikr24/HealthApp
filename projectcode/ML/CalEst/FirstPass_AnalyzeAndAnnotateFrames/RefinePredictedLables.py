#!/usr/bin/env python3
"""
Nuanced labeling:
- INPUTS (do not modify):
    CSV : /app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/custom_yolo_detections.csv
    IMGS: /app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/*.jpg (already YOLO-annotated)

- OUTPUTS (new, under nuancedlabeling/):
    Annotated images  : .../CustomYolo/nuancedlabeling/<same frame names>
    CSV (hands only)  : .../CustomYolo/nuancedlabeling/hands_left_right.csv
    CSV (all + side)  : .../CustomYolo/nuancedlabeling/all_detections_with_hand_side.csv

Rules:
  - If >=2 hands visible in a frame: leftmost -> left_hand, rightmost -> right_hand (ignore mid-band).
  - If exactly 1 hand: assign by center vs midline unless the hand box intersects the mid-band; if it intersects, label 'unknown' (skip).
"""

from pathlib import Path
import csv
import cv2

# ---- PATHS -------------------------------------------------------------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo")
CSV_PATH = BASE / "custom_yolo_detections.csv"
IMAGES_DIR = BASE  # images already drawn by YOLO live here
OUT_DIR = BASE / "nuancedlabeling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- CONFIG ------------------------------------------------------------
# classes considered "hands"
HAND_CLASSNAMES = {"hand", "hand_open", "hand_closed", "hand_pinch", "palm", "on_palm"}

# the “uncertain” band around midline used ONLY for the single-hand case (fraction of width)
MID_BAND_FRAC = 0.06

# drawing
LINE_THICK = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
TXT_SCALE = 0.5
TXT_THICK = 1

def color_for_side(side):
    # BGR
    return {
        "left":    (40, 180, 40),   # green
        "right":   (40, 40, 200),   # red-ish
        "unknown": (180, 180, 40),  # yellow-ish
        "other":   (200, 200, 200), # gray
    }.get(side, (200, 200, 200))

def intersects_mid_band(x1, x2, mid_x, band_halfw):
    band_l = mid_x - band_halfw
    band_r = mid_x + band_halfw
    return not (x2 < band_l or x1 > band_r)

def load_detections_by_frame(csv_path: Path):
    frames = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["img_w"] = int(row["img_w"]); row["img_h"] = int(row["img_h"])
                row["cls_id"] = int(row["cls_id"]) if row.get("cls_id","") != "" else -1
                row["conf"] = float(row["conf"]) if row.get("conf","") != "" else 0.0
                row["x1"] = float(row["x1"]); row["y1"] = float(row["y1"])
                row["x2"] = float(row["x2"]); row["y2"] = float(row["y2"])
            except Exception:
                continue
            frames.setdefault(row["frame"], []).append(row)
    return frames

def assign_hand_sides_for_frame(dets, img_w):
    """
    Returns: dict[index -> 'left'|'right'|'unknown'|'not_hand']
    Policy:
      - Two or more hands: force-assign leftmost and rightmost (ignore band). Others use center vs midline with band.
      - One hand: use band; if intersects mid-band -> 'unknown', else left/right by center.
    """
    side_map = {i: "not_hand" for i in range(len(dets))}
    hand_idxs = [i for i, d in enumerate(dets) if d["cls_name"].lower() in HAND_CLASSNAMES]
    if not hand_idxs:
        return side_map

    mid_x = img_w / 2.0
    band_halfw = (img_w * MID_BAND_FRAC) / 2.0

    hands = []
    for i in hand_idxs:
        d = dets[i]
        cx = (d["x1"] + d["x2"]) / 2.0
        hands.append((i, cx, d["x1"], d["x2"]))

    if len(hands) >= 2:
        # extremes: force assign
        hands_sorted = sorted(hands, key=lambda t: t[1])  # by center x
        left_i, left_cx, left_x1, left_x2 = hands_sorted[0]
        right_i, right_cx, right_x1, right_x2 = hands_sorted[-1]
        side_map[left_i] = "left"
        side_map[right_i] = "right"
        # middle ones (if any): assign conservatively
        for i, cx, x1, x2 in hands_sorted[1:-1]:
            if intersects_mid_band(x1, x2, mid_x, band_halfw):
                side_map[i] = "unknown"
            else:
                side_map[i] = "left" if cx < mid_x else "right"
    else:
        # exactly one hand
        i, cx, x1, x2 = hands[0]
        if intersects_mid_band(x1, x2, mid_x, band_halfw):
            side_map[i] = "unknown"
        else:
            side_map[i] = "left" if cx < mid_x else "right"

    return side_map

def main():
    frames = load_detections_by_frame(CSV_PATH)
    if not frames:
        print(f"[ERROR] No detections loaded from {CSV_PATH}")
        return

    hands_rows = []
    all_rows_with_side = []

    for fname, dets in frames.items():
        # open the ALREADY-ANNOTATED image as background
        img_path = IMAGES_DIR / fname
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Unreadable image: {img_path}")
            continue

        H, W = img.shape[:2]
        mid_x = W / 2.0
        band_halfw = (W * MID_BAND_FRAC) / 2.0

        # side assignment (does NOT modify original CSV)
        side_map = assign_hand_sides_for_frame(dets, W)

        # draw midline and band on top of the YOLO-rendered image
        mid_x_int = int(round(mid_x))
        cv2.line(img, (mid_x_int, 0), (mid_x_int, H), (200, 200, 200), LINE_THICK)
        band_l = int(round(mid_x - band_halfw))
        band_r = int(round(mid_x + band_halfw))
        cv2.line(img, (band_l, 0), (band_l, H), (160, 160, 160), 1)
        cv2.line(img, (band_r, 0), (band_r, H), (160, 160, 160), 1)
        cv2.putText(img, "midline", (mid_x_int + 6, 18), FONT, TXT_SCALE, (200, 200, 200), TXT_THICK, cv2.LINE_AA)

        # overlay ONLY the hand-side labels (we don't redraw boxes to avoid clutter)
        # but we will draw a small tag at the top-left of the original box.
        for i, d in enumerate(dets):
            side = side_map[i]
            cls = d["cls_name"].lower()
            if cls not in HAND_CLASSNAMES:
                # still write to all_detections_with_hand_side.csv with blank hand_side
                all_rows_with_side.append({
                    "frame": fname, "img_w": W, "img_h": H,
                    "cls_id": d["cls_id"], "cls_name": d["cls_name"], "conf": d["conf"],
                    "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                    "hand_side": ""
                })
                continue

            # CSV rows
            all_rows_with_side.append({
                "frame": fname, "img_w": W, "img_h": H,
                "cls_id": d["cls_id"], "cls_name": d["cls_name"], "conf": d["conf"],
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "hand_side": side
            })

            if side in ("left", "right"):
                hands_rows.append({
                    "frame": fname, "img_w": W, "img_h": H,
                    "cls_name": "left_hand" if side == "left" else "right_hand",
                    "conf": d["conf"],
                    "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                    "side": side,
                    "reason": "two+_hands_extremes" if sum(1 for s in side_map.values() if s in ("left","right")) >= 2
                              else "single_hand_center_rule"
                })

            # draw a small side tag on the image (non-destructive overlay)
            tag = {"left": "LEFT", "right": "RIGHT", "unknown": "HAND?"}.get(side, "")
            if tag:
                x1, y1 = int(d["x1"]), int(d["y1"])
                cv2.putText(img, tag, (x1, max(15, y1 - 6)), FONT, TXT_SCALE,
                            color_for_side(side if side in ("left","right","unknown") else "other"),
                            TXT_THICK, cv2.LINE_AA)

        # save nuanced image
        out_img = OUT_DIR / fname
        cv2.imwrite(str(out_img), img)

    # write CSVs (brand-new; original CSV is untouched)
    hands_csv = OUT_DIR / "hands_left_right.csv"
    all_csv = OUT_DIR / "all_detections_with_hand_side.csv"

    if hands_rows:
        with open(hands_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame","img_w","img_h","cls_name","conf","x1","y1","x2","y2","side","reason"
            ])
            writer.writeheader()
            writer.writerows(hands_rows)
        print(f"[OK] Wrote: {hands_csv} ({len(hands_rows)} rows)")
    else:
        print("[WARN] No left/right hands to write (all single-hand near midline?).")

    if all_rows_with_side:
        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame","img_w","img_h","cls_id","cls_name","conf","x1","y1","x2","y2","hand_side"
            ])
            writer.writeheader()
            writer.writerows(all_rows_with_side)
        print(f"[OK] Wrote: {all_csv} ({len(all_rows_with_side)} rows)")

    print(f"[DONE] Images saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
