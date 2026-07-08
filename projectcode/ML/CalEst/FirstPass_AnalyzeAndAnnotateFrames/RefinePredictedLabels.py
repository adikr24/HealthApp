#!/usr/bin/env python3
from pathlib import Path
import csv
import cv2
## This code essentially assigns left-right hands to the predicted hand labels to further compute hand-object distance
# ---- PATHS -------------------------------------------------------------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo")
CSV_PATH = BASE / "custom_yolo_detections.csv"
IMAGES_DIR = BASE
OUT_DIR = BASE / "detailed_hand_sides"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- CONFIG ------------------------------------------------------------
# Treat these as "hand" (normalize to 'hand'). You listed: hand_open, hand, hand_fist, palm.
# We also keep prior variants for safety.
_HAND_EQUIV = {
    "hand", "hand_open", "hand_fist", "palm",
    "hand_closed", "hand_pinch", "on_palm",  # legacy/nearby variants if they appear
}
# Optional alias mapping if you want a single canonical class in outputs
def normalize_hand_name(name: str) -> str:
    n = (name or "").casefold()
    return "hand" if n in _HAND_EQUIV else n

# The “uncertain” band around midline for single-hand case (fraction of width)
MID_BAND_FRAC = 0.06

# Drawing
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
                row["frame"]  = row["frame"]
                row["img_w"]  = int(float(row["img_w"]))
                row["img_h"]  = int(float(row["img_h"]))
                row["cls_id"] = int(row["cls_id"]) if row.get("cls_id","") != "" else -1
                row["conf"]   = float(row["conf"]) if row.get("conf","") != "" else 0.0
                row["x1"]     = float(row["x1"]); row["y1"] = float(row["y1"])
                row["x2"]     = float(row["x2"]); row["y2"] = float(row["y2"])
                # store originals + normalized
                row["cls_name_orig"] = row.get("cls_name","")
                row["cls_name"]      = (row.get("cls_name") or "").casefold()
                row["cls_name_norm"] = normalize_hand_name(row.get("cls_name"))
            except Exception:
                continue
            frames.setdefault(row["frame"], []).append(row)
    return frames

def is_hand_row(row) -> bool:
    return row.get("cls_name","") in _HAND_EQUIV or row.get("cls_name_norm","") == "hand"

def assign_hand_sides_for_frame(dets, img_w):
    """
    Returns: dict[index -> 'left'|'right'|'unknown'|'not_hand']

    Policy:
      - 2+ hands: leftmost -> left, rightmost -> right (ignore band); middle hands use band rule.
      - 1 hand: use band; if its box intersects the mid-band => 'unknown', else left/right by center.
    """
    side_map = {i: "not_hand" for i in range(len(dets))}
    hand_idxs = [i for i, d in enumerate(dets) if is_hand_row(d)]
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
        left_i, _, _, _  = hands_sorted[0]
        right_i, _, _, _ = hands_sorted[-1]
        side_map[left_i]  = "left"
        side_map[right_i] = "right"
        # middle ones (if any): assign conservatively with band
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

        side_map = assign_hand_sides_for_frame(dets, W)

        # draw midline + band
        mid_x_int = int(round(mid_x))
        cv2.line(img, (mid_x_int, 0), (mid_x_int, H), (200, 200, 200), LINE_THICK)
        band_l = int(round(mid_x - band_halfw))
        band_r = int(round(mid_x + band_halfw))
        cv2.line(img, (band_l, 0), (band_l, H), (160, 160, 160), 1)
        cv2.line(img, (band_r, 0), (band_r, H), (160, 160, 160), 1)
        cv2.putText(img, "midline", (mid_x_int + 6, 18), FONT, TXT_SCALE, (200, 200, 200), TXT_THICK, cv2.LINE_AA)

        # overlay side tags + prepare CSV rows
        for i, d in enumerate(dets):
            side = side_map[i]

            # Always write all_detections_with_hand_side
            all_rows_with_side.append({
                "frame": fname, "img_w": W, "img_h": H,
                "cls_id": d["cls_id"],
                "cls_name_orig": d.get("cls_name_orig",""),
                "cls_name_norm": d.get("cls_name_norm",""),
                "conf": d["conf"],
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "hand_side": side if is_hand_row(d) else ""
            })

            # Hands-only CSV
            if is_hand_row(d):
                if side in ("left", "right"):
                    reason = "two+_hands_extremes" if sum(1 for s in side_map.values() if s in ("left","right")) >= 2 \
                             else "single_hand_center_rule"
                else:
                    reason = "midband_ambiguous"

                hands_rows.append({
                    "frame": fname, "img_w": W, "img_h": H,
                    "cls_name_orig": d.get("cls_name_orig",""),
                    "cls_name_norm": d.get("cls_name_norm",""),
                    "conf": d["conf"],
                    "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                    "side": side,
                    "reason": reason
                })

                # draw tag for hands
                tag = {"left": "LEFT", "right": "RIGHT", "unknown": "HAND?"}.get(side, "")
                if tag:
                    x1, y1 = int(d["x1"]), int(d["y1"])
                    cv2.putText(img, tag, (x1, max(15, y1 - 6)), FONT, TXT_SCALE,
                                color_for_side(side if side in ("left","right","unknown") else "other"),
                                TXT_THICK, cv2.LINE_AA)

        # save nuanced image
        out_img = OUT_DIR / fname
        cv2.imwrite(str(out_img), img)

    # write CSVs
    hands_csv = OUT_DIR / "hands_left_right.csv"
    all_csv   = OUT_DIR / "all_detections_with_hand_side.csv"

    if hands_rows:
        with open(hands_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame","img_w","img_h","cls_name_orig","cls_name_norm","conf","x1","y1","x2","y2","side","reason"
            ])
            writer.writeheader()
            writer.writerows(hands_rows)
        print(f"[OK] Wrote: {hands_csv} ({len(hands_rows)} rows)")
    else:
        print("[WARN] No hand rows found or all were 'unknown' near midline.")

    if all_rows_with_side:
        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame","img_w","img_h","cls_id","cls_name_orig","cls_name_norm","conf","x1","y1","x2","y2","hand_side"
            ])
            writer.writeheader()
            writer.writerows(all_rows_with_side)
        print(f"[OK] Wrote: {all_csv} ({len(all_rows_with_side)} rows)")

    print(f"[DONE] Images saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
