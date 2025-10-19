#!/usr/bin/env python3
# compute_object_to_voi_distance.py
# Detect ANY object that comes near or crosses the VOI ROI (e.g., blender mouth)
# and record start/end frames of activity + distances.
# Uses the merged VOI file from the previous step (voi_bbox_and_hands_labels.csv).
# Draws annotated frames from VOIBoundingBoxes images.

from pathlib import Path
import csv, re, math, cv2
from collections import defaultdict

# ======================= CONFIG =======================

# ---- INPUTS ----
IN_LABELS = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/voi_bbox_and_hands_labels.csv"
)
IN_IMG_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIBoundingBoxes"
)

# ---- OUTPUTS ----
OUT_META_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata"
)
OUT_IMG_DIR = Path(
    "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIActivity"
)

OUT_CSV = OUT_META_DIR / "voi_activity_summary.csv"

# ---- PARAMETERS ----
ROI_NAME = "blender_mouth"
NEAR_THRESHOLD_PX = 80          # pixel distance threshold for "near ROI"
GAP_TOLERANCE = 10              # tolerate this many missing frames before ending activity
MIN_EVENT_FRAMES = 3            # ignore shorter blips
OVERLAP_IS_ACTIVITY = True      # if bbox overlaps ROI bbox, treat as activity
MIN_IOU_FOR_OVERLAP = 0.0       # >0 means any overlap; raise if you want stricter

# ---- VISUALIZATION ----
WRITE_ANNOTATIONS = True
ROI_COLOR = (0, 128, 255)
OBJ_COLOR = (255, 0, 0)
LINE_COLOR = (0, 0, 255)
OVERLAP_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICK = 2

# ======================= HELPERS =======================

def norm(s): 
    return (s or "").strip().lower()

def frame_idx(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else -1

def diag(w, h):
    try: return math.hypot(float(w), float(h))
    except: return None

def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def rect_intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, (0, 0, 0, 0)
    return float((x2 - x1) * (y2 - y1)), (x1, y1, x2, y2)

def rect_area(a):
    x1, y1, x2, y2 = a
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou(a, b):
    inter, _ = rect_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    ua = rect_area(a) + rect_area(b) - inter
    return inter / max(ua, 1e-6)

def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if WRITE_ANNOTATIONS:
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
    Returns dict[frame] = {'roi': (cx,cy,rx1,ry1,rx2,ry2), 'objects':[...]}
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

        # ROI (from blender rows)
        if norm(r.get("roi_name", "")) == ROI_NAME:
            try:
                rcx = float(r.get("roi_cx", "")); rcy = float(r.get("roi_cy", ""))
                rx1 = float(r.get("roi_x1", "")); ry1 = float(r.get("roi_y1", ""))
                rx2 = float(r.get("roi_x2", "")); ry2 = float(r.get("roi_y2", ""))
                frames[frame]["roi"] = (rcx, rcy, rx1, ry1, rx2, ry2)
            except:
                pass
        else:
            frames[frame]["objects"].append({
                "cls_name": cls,
                "box": (x1, y1, x2, y2)
            })

    return dict(sorted(frames.items(), key=lambda kv: frame_idx(kv[0])))

# ======================= ACTIVITY TRACKING =======================

def main():
    ensure_dirs()
    rows = load_labels()
    frames = build_frame_dict(rows)
    print(f"[INFO] Loaded {len(frames)} frames from {IN_LABELS.name}")

    active = {}
    events = []
    img_cap = 0

    for fname, data in frames.items():
        roi = data["roi"]
        if not roi:
            continue

        rcx, rcy, rx1, ry1, rx2, ry2 = roi
        roi_center = (rcx, rcy)
        roi_box = (rx1, ry1, rx2, ry2)
        objs = data["objects"]

        img = None
        if WRITE_ANNOTATIONS:
            img_path = IN_IMG_DIR / fname
            if img_path.exists():
                img = cv2.imread(str(img_path))

        for o in objs:
            cls = o["cls_name"]
            box = o["box"]

            oc = bbox_center(box)
            d = distance(oc, roi_center)
            ov_iou = iou(box, roi_box) if OVERLAP_IS_ACTIVITY else 0.0
            overlaps = OVERLAP_IS_ACTIVITY and (ov_iou > MIN_IOU_FOR_OVERLAP)
            near = (d <= NEAR_THRESHOLD_PX) or overlaps
            fid = frame_idx(fname)

            # --- track object activity ---
            if near:
                if cls not in active:
                    active[cls] = {
                        "start_frame": fname,
                        "start_idx": fid,
                        "last_frame": fname,
                        "last_idx": fid,
                        "distances": [d],
                        "overlap_hits": 1 if overlaps else 0,
                        "missing": 0
                    }
                else:
                    ev = active[cls]
                    ev["last_frame"] = fname
                    ev["last_idx"] = fid
                    ev["distances"].append(d)
                    if overlaps:
                        ev["overlap_hits"] += 1
                    ev["missing"] = 0
            else:
                if cls in active:
                    active[cls]["missing"] += 1
                    if active[cls]["missing"] > GAP_TOLERANCE:
                        ev = active.pop(cls)
                        dur = ev["last_idx"] - ev["start_idx"] + 1
                        if dur >= MIN_EVENT_FRAMES:
                            events.append({
                                "class": cls,
                                "start_frame": ev["start_frame"],
                                "end_frame": ev["last_frame"],
                                "start_idx": ev["start_idx"],
                                "end_idx": ev["last_idx"],
                                "avg_dist": round(sum(ev["distances"]) / len(ev["distances"]), 1),
                                "min_dist": round(min(ev["distances"]), 1),
                                "overlap_hits": ev.get("overlap_hits", 0)
                            })

            # --- Visualization ---
            if WRITE_ANNOTATIONS and img is not None:
                cv2.rectangle(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)), ROI_COLOR, 2)
                cv2.circle(img, (int(rcx), int(rcy)), 4, ROI_COLOR, -1)

                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), OBJ_COLOR, 2)
                cv2.circle(img, (int(oc[0]), int(oc[1])), 3, OBJ_COLOR, -1)
                cv2.line(img, (int(oc[0]), int(oc[1])), (int(rcx), int(rcy)),
                         OVERLAP_COLOR if overlaps else LINE_COLOR, 2)
                label = f"{cls}:{int(d)}px"
                if overlaps:
                    label += " (overlap)"
                cv2.putText(img, label, (int(bx1), max(0, int(by1) - 6)), FONT, 0.5,
                            OVERLAP_COLOR if overlaps else LINE_COLOR, 1)

        if WRITE_ANNOTATIONS and img is not None:
            cv2.imwrite(str(OUT_IMG_DIR / fname), img)
            img_cap += 1

    # finalize still-active events
    for cls, ev in active.items():
        dur = ev["last_idx"] - ev["start_idx"] + 1
        if dur >= MIN_EVENT_FRAMES:
            events.append({
                "class": cls,
                "start_frame": ev["start_frame"],
                "end_frame": ev["last_frame"],
                "start_idx": ev["start_idx"],
                "end_idx": ev["last_idx"],
                "avg_dist": round(sum(ev["distances"]) / len(ev["distances"]), 1),
                "min_dist": round(min(ev["distances"]), 1),
                "overlap_hits": ev.get("overlap_hits", 0)
            })

    # Write events CSV
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "class","start_frame","end_frame","start_idx","end_idx",
            "avg_dist","min_dist","overlap_hits"
        ])
        writer.writeheader()
        writer.writerows(events)


    print(f"[DONE] Activity events written → {OUT_CSV} ({len(events)} events)")
    if WRITE_ANNOTATIONS:
        print(f"[DONE] Annotated frames → {OUT_IMG_DIR} ({img_cap} frames)")

if __name__ == "__main__":
    main()
