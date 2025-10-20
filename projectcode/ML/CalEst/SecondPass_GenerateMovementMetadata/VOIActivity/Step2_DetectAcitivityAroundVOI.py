#!/usr/bin/env python3
# detectactivityaroundvoi.py (drop-in)
# Detect ANY object near the VOI mouth ROI and summarize contiguous activity windows.
# Adds robust handling for large frame-number jumps and per-object index gaps.

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
GAP_TOLERANCE = 10              # tolerate this many consecutive non-near frames before ending activity
MIN_EVENT_FRAMES = 3            # ignore shorter blips

OVERLAP_IS_ACTIVITY = True      # if bbox overlaps ROI bbox, treat as activity
MIN_IOU_FOR_OVERLAP = 0.0       # >0 means any overlap; raise for stricter overlap

# ---- NEW: frame-jump handling ----
# You said you skip ~10 frames, and want to drop continuity if name gap > 20.
FRAME_JUMP_CLOSE_ALL = 20       # if current_fid - prev_fid > this, close ALL active events
PER_OBJECT_MAX_INDEX_GAP = 20   # if current_fid - last_seen_fid(object) > this, close that object's event

# ---- VISUALIZATION ----
WRITE_ANNOTATIONS = True
ROI_COLOR = (0, 128, 255)
OBJ_COLOR = (255, 0, 0)
LINE_COLOR = (0, 0, 255)
OVERLAP_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
THICK = 2

# ---- LOGGING ----
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
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)
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
    Returns dict[frame] = {'roi': (cx,cy,rx1,ry1,rx2,ry2), 'objects':[{'cls_name', 'box'}]}
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

def _close_event_for_cls(cls, ev, events):
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
        if DEBUG:
            print(f"[DEBUG] Closed event: {cls} {ev['start_idx']}→{ev['last_idx']} "
                  f"(avg_dist={round(sum(ev['distances'])/len(ev['distances']),1)}, "
                  f"overlap_hits={ev.get('overlap_hits',0)})")

def main():
    ensure_dirs()
    rows = load_labels()
    frames = build_frame_dict(rows)
    print(f"[INFO] Loaded {len(frames)} frames from {IN_LABELS.name}")

    active = {}
    events = []
    img_cap = 0
    prev_fid = None

    for fname, data in frames.items():
        fid = frame_idx(fname)

        # ---------- CLOSE ALL on large discontinuity ----------
        if prev_fid is not None and (fid - prev_fid) > FRAME_JUMP_CLOSE_ALL:
            if DEBUG:
                print(f"[DEBUG] Large frame jump detected: {prev_fid} → {fid} (> {FRAME_JUMP_CLOSE_ALL}) "
                      f"→ closing ALL active events")
            for cls, ev in list(active.items()):
                _close_event_for_cls(cls, ev, events)
            active.clear()
        prev_fid = fid
        # -------------------------------------------------------

        # ---------- GLOBAL GAP CHECK for all active classes ----------
        for cls, ev in list(active.items()):
            gap_since_last_seen = fid - ev["last_idx"]
            if gap_since_last_seen > PER_OBJECT_MAX_INDEX_GAP:
                if DEBUG:
                    print(f"[DEBUG] Force close {cls} (global gap): "
                          f"{ev['last_idx']} → {fid} (gap {gap_since_last_seen})")
                _close_event_for_cls(cls, ev, events)
                active.pop(cls, None)
        # -------------------------------------------------------------

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

        # --------- process detected objects ----------
        for o in objs:
            cls = o["cls_name"]
            box = o["box"]

            oc = bbox_center(box)
            d = distance(oc, roi_center)
            ov_iou = iou(box, roi_box) if OVERLAP_IS_ACTIVITY else 0.0
            overlaps = OVERLAP_IS_ACTIVITY and (ov_iou > MIN_IOU_FOR_OVERLAP)
            near = (d <= NEAR_THRESHOLD_PX) or overlaps

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
                    if DEBUG:
                        print(f"[DEBUG] Start event: {cls} @ {fid}")
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
                    ev = active[cls]
                    ev["missing"] += 1
                    gap_since_last_seen = fid - ev["last_idx"]

                    if ev["missing"] > GAP_TOLERANCE or gap_since_last_seen > PER_OBJECT_MAX_INDEX_GAP:
                        if DEBUG:
                            print(f"[DEBUG] Close {cls} (local gap): "
                                  f"{ev['last_idx']} → {fid} (gap {gap_since_last_seen})")
                        _close_event_for_cls(cls, ev, events)
                        active.pop(cls, None)

        # --------- handle objects not seen in this frame ----------
        seen_classes = {o["cls_name"] for o in objs}
        for cls, ev in list(active.items()):
            if cls not in seen_classes:
                ev["missing"] += 1
                gap_since_last_seen = fid - ev["last_idx"]
                if ev["missing"] > GAP_TOLERANCE or gap_since_last_seen > PER_OBJECT_MAX_INDEX_GAP:
                    if DEBUG:
                        print(f"[DEBUG] Close {cls} (not seen): "
                              f"{ev['last_idx']} → {fid} (gap {gap_since_last_seen})")
                    _close_event_for_cls(cls, ev, events)
                    active.pop(cls, None)
        # -----------------------------------------------------------

        if WRITE_ANNOTATIONS and img is not None:
            cv2.imwrite(str(OUT_IMG_DIR / fname), img)
            img_cap += 1

    # finalize still-active events
    for cls, ev in list(active.items()):
        _close_event_for_cls(cls, ev, events)

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
