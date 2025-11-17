#!/usr/bin/env python3
# visualize_voi_nearest_nonhand.py
#
# For each frame:
#   - load VOI (ROI) center & box
#   - load all object boxes
#   - ignore hands
#   - find the nearest non-hand object to VOI center
#   - optionally, if distance <= NEAR_THRESHOLD_PX, draw a line + distance text
#
# Outputs annotated images to OUT_IMG_DIR.

from pathlib import Path
import csv, re, math, cv2
from collections import defaultdict

# ======================= CONFIG =======================

# ---- INPUTS ----
IN_LABELS = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/voi_bbox_and_hands_labels.csv"
)
IN_IMG_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIBoundingBoxes"
)

# ---- OUTPUTS ----
OUT_IMG_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIActivity_DistViz"
)

ROI_NAME = "blender"

# Only draw line for nearest object if it's this close (in pixels)
NEAR_THRESHOLD_PX = 50

# Classes to ignore when searching for nearest object
HAND_CLASSES = {"hand", "hand_fist"}

# VISUAL STYLE
WRITE_ANNOTATIONS = True
ROI_COLOR = (0, 128, 255)   # VOI bbox
ROI_CENTER_COLOR = (0, 255, 255)
OBJ_COLOR = (255, 0, 0)     # all object boxes
NEAREST_COLOR = (0, 255, 0) # highlight nearest object + line
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICK = 2

DEBUG = True

# ======================= HELPERS =======================

def norm(s):
    return (s or "").strip().lower()

def frame_idx(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else -1

def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def ensure_dirs():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ======================= LOAD LABELS =======================

def load_labels():
    if not IN_LABELS.exists():
        raise FileNotFoundError(f"Missing VOI labels file: {IN_LABELS}")
    with IN_LABELS.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows

# ======================= BUILD PER FRAME =======================

def build_frame_dict(rows):
    """
    Returns dict[frame] = {
        'roi': (cx, cy, rx1, ry1, rx2, ry2) or None,
        'objects': [{'cls_name', 'box'}]
    }
    """
    frames = defaultdict(lambda: {"roi": None, "objects": []})

    for r in rows:
        frame = Path(r.get("frame") or "").name
        if not frame:
            continue

        cls = norm(r.get("cls_name_norm") or r.get("cls_name") or r.get("cls_name_orig") or "")

        try:
            x1, y1, x2, y2 = map(float, [r["x1"], r["y1"], r["x2"], r["y2"]])
        except Exception:
            continue

        # ROI row
        if norm(r.get("roi_name", "")) == ROI_NAME:
            try:
                rcx = float(r.get("roi_cx", "")); rcy = float(r.get("roi_cy", ""))
                rx1 = float(r.get("roi_x1", "")); ry1 = float(r.get("roi_y1", ""))
                rx2 = float(r.get("roi_x2", "")); ry2 = float(r.get("roi_y2", ""))
                frames[frame]["roi"] = (rcx, rcy, rx1, ry1, rx2, ry2)
            except Exception:
                pass
        else:
            frames[frame]["objects"].append({
                "cls_name": cls,
                "box": (x1, y1, x2, y2)
            })

    # sort by frame index
    return dict(sorted(frames.items(), key=lambda kv: frame_idx(kv[0])))

# ======================= MAIN VIZ LOOP =======================

def main():
    ensure_dirs()
    rows = load_labels()
    frames = build_frame_dict(rows)
    print(f"[INFO] Loaded {len(frames)} frames from {IN_LABELS.name}")

    total_objects = 0
    frames_with_nearest = 0

    for fname, data in frames.items():
        roi = data["roi"]
        objs = data["objects"]

        if not roi:
            continue  # no VOI on this frame

        rcx, rcy, rx1, ry1, rx2, ry2 = roi
        roi_center = (rcx, rcy)

        img_path = IN_IMG_DIR / fname
        if not img_path.exists():
            if DEBUG:
                print(f"[WARN] Image not found for frame {fname}: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            if DEBUG:
                print(f"[WARN] Failed to read image for frame {fname}: {img_path}")
            continue

        # Draw VOI bbox + center marker
        cv2.rectangle(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), ROI_COLOR, THICK)
        cv2.circle(img, (int(rcx), int(rcy)), 4, ROI_CENTER_COLOR, -1)
        cv2.putText(img, "VOI", (int(rx1), int(ry1) - 5), FONT, 0.6, ROI_COLOR, 2, cv2.LINE_AA)

        # Draw all objects (for context)
        for o in objs:
            total_objects += 1
            box = o["box"]
            cls = o["cls_name"]

            cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                OBJ_COLOR,
                1
            )
            cv2.putText(
                img,
                cls,
                (int(box[0]), int(box[1]) - 3),
                FONT,
                0.5,
                OBJ_COLOR,
                1,
                cv2.LINE_AA
            )

        # ---- find nearest NON-HAND object ----
        # ---- find nearest NON-HAND object ----
        non_hand_objs = [o for o in objs if o["cls_name"] not in HAND_CLASSES]

        if non_hand_objs:
            # compute distances
            dists = []
            for o in non_hand_objs:
                ocx, ocy = bbox_center(o["box"])
                d = distance((ocx, ocy), roi_center)
                dists.append((d, o, (ocx, ocy)))

            # nearest by distance
            min_d, nearest_obj, nearest_center = min(dists, key=lambda t: t[0])
            frames_with_nearest += 1

            if DEBUG:
                print(f"[DEBUG] {fname}: nearest non-hand = {nearest_obj['cls_name']}, "
                      f"dist={min_d:.2f}px")

            ocx, ocy = nearest_center
            bx1, by1, bx2, by2 = nearest_obj["box"]

            # highlight nearest object's box
            cv2.rectangle(
                img,
                (int(bx1), int(by1)),
                (int(bx2), int(by2)),
                NEAREST_COLOR,
                2
            )

            # draw line VOI center -> nearest object center
            cv2.line(
                img,
                (int(rcx), int(rcy)),
                (int(ocx), int(ocy)),
                NEAREST_COLOR,
                2
            )

            # distance label at line midpoint (ALWAYS drawn)
            mid_x = int((rcx + ocx) / 2.0)
            mid_y = int((rcy + ocy) / 2.0)
            cv2.putText(
                img,
                f"{min_d:.1f}",
                (mid_x, mid_y),
                FONT,
                0.7,
                NEAREST_COLOR,
                2,
                cv2.LINE_AA
            )

        # Save annotated image
        out_path = OUT_IMG_DIR / fname
        cv2.imwrite(str(out_path), img)
        if DEBUG:
            print(f"[DEBUG] Wrote annotated frame: {out_path}")

    print(f"[DONE] Annotated frames → {OUT_IMG_DIR}")
    print(f"[STATS] Total objects: {total_objects}, frames with non-hand nearest: {frames_with_nearest}")
    print(f"[PARAM] NEAR_THRESHOLD_PX = {NEAR_THRESHOLD_PX}")


if __name__ == "__main__":
    main()
