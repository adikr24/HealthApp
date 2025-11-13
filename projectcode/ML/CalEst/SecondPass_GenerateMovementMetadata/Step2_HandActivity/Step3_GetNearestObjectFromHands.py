#!/usr/bin/env python3
# draw_nearest_object_per_hand_clean.py

import csv
import math
from pathlib import Path
import cv2


# ======= PATHS ===============================================================
META_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/HandActivity")
LABELS_CSV = META_DIR / "hand_rois.csv"

HAND_ROI_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/HandROIs")
OUT_DIR      = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/HandNearestObjects")

OUT_DIR.mkdir(parents=True, exist_ok=True)

NEAREST_CSV = META_DIR / "hand_roi_closest_object.csv"
NO_OUTPUT_SUMMARY = META_DIR / "hand_roi_no_object_detection_summary.csv"
# ============================================================================

# Hand colors (BGR)
HAND_COLORS = {
    "left":    (0, 255, 0),     # green
    "right":   (0, 165, 255),   # orange
    "merged":  (255, 255, 0),   # cyan-ish
    "unknown": (255, 0, 255),   # magenta
}

def f2(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def classify_hand_side(label: str, side: str):
    label = (label or "").strip().lower()
    side  = (side or "").strip().lower()
    if side in ("left", "right"): return side
    if "left" in label: return "left"
    if "right" in label: return "right"
    if "merged" in label: return "merged"
    return "unknown"

def read_all_rows(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))

def frames_in_rows(rows):
    return sorted({(r.get("frame") or "").strip() for r in rows if (r.get("frame") or "").strip()})

def rows_for_frame(rows, frame_name):
    return [r for r in rows if (r.get("frame") or "").strip() == frame_name]

def collect_hands(rows):
    hands = []
    for r in rows:
        label = (r.get("label") or "")
        if "hand" not in label.lower():
            continue
        hcx, hcy = f2(r.get("hand_roi_cx")), f2(r.get("hand_roi_cy"))
        if hcx is None or hcy is None:
            continue
        hx1, hy1 = f2(r.get("hand_roi_x1")), f2(r.get("hand_roi_y1"))
        hx2, hy2 = f2(r.get("hand_roi_x2")), f2(r.get("hand_roi_y2"))
        side = classify_hand_side(label, r.get("hand_side"))
        tag  = {"left":"L","right":"R","merged":"M"}.get(side,"U")
        hands.append({
            "label": label.strip(),
            "side": side,
            "tag": tag,
            "bbox": (hx1, hy1, hx2, hy2),
            "centroid": (hcx, hcy),
        })
    return hands

def collect_objects(rows):
    objs = []
    for r in rows:
        name = (r.get("cls_name_norm") or r.get("cls_name_orig") or "").strip().lower()
        if not name:
            continue
        x1, y1, x2, y2 = f2(r.get("x1")), f2(r.get("y1")), f2(r.get("x2")), f2(r.get("y2"))
        if None in (x1, y1, x2, y2):
            continue
        cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
        conf = r.get("conf")
        try:
            conf_num = float(conf) if conf not in (None, "") else None
        except:
            conf_num = None
        objs.append({
            "name": name,
            "bbox": (x1, y1, x2, y2),
            "centroid": (cx, cy),
            "conf": conf,
            "conf_num": conf_num
        })
    return objs

def pick_nearest_per_hand(hands, objects, diag_len):
    results = []
    for h in hands:
        (hcx, hcy) = h["centroid"]
        best = None
        for obj in objects:
            (cx, cy) = obj["centroid"]
            dist = math.hypot(cx - hcx, cy - hcy)
            frac = dist / diag_len if diag_len and diag_len > 0 else 0.0
            cand = {"hand": h, "object": obj, "dist_px": dist, "frac": frac}
            if best is None:
                best = cand
            else:
                if cand["dist_px"] < best["dist_px"] - 1e-6:
                    best = cand
                elif abs(cand["dist_px"] - best["dist_px"]) <= 1e-6:
                    c1, c2 = obj.get("conf_num"), best["object"].get("conf_num")
                    if (c1 is not None and c2 is not None and c1 > c2) or (c1 is not None and c2 is None):
                        best = cand
        if best:
            results.append(best)
    return results

def draw_frame(img_path, out_path, hands, objects, nearest_pairs, img_w, img_h):
    img = cv2.imread(str(img_path))
    if img is None:
        return "missing_image"
    H, W = img.shape[:2]
    W = int(img_w) if img_w else W
    H = int(img_h) if img_h else H

    # draw objects
    for obj in objects:
        (x1,y1,x2,y2) = obj["bbox"]
        (cx,cy) = obj["centroid"]
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
        cv2.circle(img, (int(cx),int(cy)), 4, (255,0,0), -1)
        txt = obj["name"]
        if obj["conf"] not in (None, ""):
            try: txt += f" ({float(obj['conf']):.2f})"
            except: txt += f" ({obj['conf']})"
        cv2.putText(img, txt, (int(x1), max(0, int(y1)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

    # draw hands
    for h in hands:
        (hx1,hy1,hx2,hy2) = h["bbox"]
        (hcx,hcy) = h["centroid"]
        color = HAND_COLORS.get(h["side"], HAND_COLORS["unknown"])
        if None not in (hx1,hy1,hx2,hy2):
            cv2.rectangle(img, (int(hx1),int(hy1)), (int(hx2),int(hy2)), color, 2)
        cv2.circle(img, (int(hcx),int(hcy)), 5, color, -1)
        cv2.putText(img, h["label"], (int(hcx)+6, int(hcy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # draw nearest lines
    for pair in nearest_pairs:
        h = pair["hand"]; obj = pair["object"]
        (hcx,hcy) = h["centroid"]; (cx,cy) = obj["centroid"]
        color = HAND_COLORS.get(h["side"], HAND_COLORS["unknown"])
        cv2.line(img, (int(hcx),int(hcy)), (int(cx),int(cy)), color, 2)
        midx, midy = int((hcx+cx)/2), int((hcy+cy)/2)
        cv2.putText(img, f"{h['tag']}→{obj['name']}: {pair['dist_px']:.1f}px ({pair['frac']:.3f})",
                    (midx, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return "ok"

def main():
    rows = read_all_rows(LABELS_CSV)
    frames = frames_in_rows(rows)
    if not frames:
        raise RuntimeError(f"No frames found in {LABELS_CSV}")

    # Prepare writers
    with open(NEAREST_CSV, "w", newline="") as fout, \
         open(NO_OUTPUT_SUMMARY, "w", newline="") as fno:
        nearest_writer = csv.writer(fout)
        no_output_writer = csv.writer(fno)

        # nearest-per-hand CSV (no out_image column)
        nearest_writer.writerow(["frame","hand_label","hand_side","hand_tag","nearest_object","distance_px","distance_frac_diag","status"])
        # summary of frames where no output was produced
        no_output_writer.writerow(["frame","reason"])

        for frame in frames:
            frame_rows = rows_for_frame(rows, frame)
            hands = collect_hands(frame_rows)
            objs  = collect_objects(frame_rows)

            if not hands:
                no_output_writer.writerow([frame, "no_hands"])
                continue
            if not objs:
                no_output_writer.writerow([frame, "no_objects"])
                continue

            img_path = HAND_ROI_DIR / frame
            out_path = OUT_DIR / frame

            img_w = f2(frame_rows[0].get("img_w"))
            img_h = f2(frame_rows[0].get("img_h"))

            # diag for frac distances
            diag = None
            if img_w and img_h:
                diag = math.hypot(img_w, img_h)
            else:
                tmp = cv2.imread(str(img_path))
                if tmp is not None:
                    H, W = tmp.shape[:2]
                    diag = math.hypot(W, H)
                else:
                    diag = 1.0  # avoid div-by-zero if image missing (status will mark)

            nearest = pick_nearest_per_hand(hands, objs, diag if diag else 1.0)
            status = draw_frame(img_path, out_path, hands, objs, nearest, img_w, img_h)

            # Write one row per hand with its nearest object
            for pair in nearest:
                nearest_writer.writerow([
                    frame,
                    pair["hand"]["label"], pair["hand"]["side"], pair["hand"]["tag"],
                    pair["object"]["name"],
                    f"{pair['dist_px']:.2f}", f"{pair['frac']:.4f}",
                    status
                ])

if __name__ == "__main__":
    main()
