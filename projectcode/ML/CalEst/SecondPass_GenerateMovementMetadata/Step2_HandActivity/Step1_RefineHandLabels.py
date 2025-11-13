#!/usr/bin/env python3
## Step 1 in Hand activity metadata generation
# hand_label_collapse.py
# - Merge overlapping hand boxes per frame (>=20% overlap).
# - Output merged boxes + annotated images + per-frame summary.
#
# Inputs:
#   /app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/detailed_hand_sides/all_detections_with_hand_side.csv
#   Images assumed alongside that CSV (same folder), e.g. frame_*.jpg|png
#
# Outputs:
#   CSV:  .../SecondPass_GenerateMovementMetadata/Metadata/hand_label_collapse.csv
#   CSV:  .../SecondPass_GenerateMovementMetadata/Metadata/hand_label_collapse_summary.csv
#   IMGs: .../SecondPass_GenerateMovementMetadata/bounding_boxes/hand_label_collapse/<frames>

from pathlib import Path
import csv, cv2, re
from collections import defaultdict

# =============== CONFIG ===============

# Source (read-only)
SRC_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/detailed_hand_sides")
IN_CSV  = SRC_DIR / "all_detections_with_hand_side.csv"

# Outputs
OUT_ROOT   = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata")

# write CSVs under Metadata/HandActivity
OUT_META   = OUT_ROOT / "Metadata" / "HandActivity"
OUT_META.mkdir(parents=True, exist_ok=True)

# images stay the same location as before
OUT_IMGS   = OUT_ROOT / "bounding_boxes" / "hand_label_collapse"
OUT_IMGS.mkdir(parents=True, exist_ok=True)

OUT_LABELS_CSV  = OUT_META / "hand_label_collapse.csv"
OUT_SUMMARY_CSV = OUT_META / "hand_label_collapse_summary.csv"


# Which classes count as "hand" in the CSV
HAND_CLASS_EQUIV = {"hand", "hand_open", "hand_fist", "palm", "hand_closed", "hand_pinch", "on_palm"}

# Merge threshold (any of these >= triggers merge)
IOU_THRESH = 0.20
INTER_OVER_MINAREA_THRESH = 0.20  # intersection / min(areaA, areaB)

# Drawing
COL_SINGLE  = (60, 200, 255)   # BGR for single left/right/unknown hand boxes
COL_MERGED  = (0, 0, 255)      # BGR for merged_hands
THICK       = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX
TSCALE      = 0.5
TTHICK      = 1

# =============== HELPERS ===============

def norm(s): return (s or "").strip().lower()

def get_cls(row):
    return norm(row.get("cls_name_norm") or row.get("cls_name") or row.get("cls_name_orig") or "")

def is_hand_row(row):
    return get_cls(row) in HAND_CLASS_EQUIV

def area(b):
    x1,y1,x2,y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def inter_rect(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    x1, y1 = max(ax1,bx1), max(ay1,by1)
    x2, y2 = min(ax2,bx2), min(ay2,by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, (0,0,0,0)
    return float((x2-x1)*(y2-y1)), (x1,y1,x2,y2)

def iou(a,b):
    inter,_ = inter_rect(a,b)
    if inter <= 0: return 0.0
    ua = area(a) + area(b) - inter
    return inter / max(ua, 1e-6)

def overlap_over_min(a,b):
    inter,_ = inter_rect(a,b)
    if inter <= 0: return 0.0
    return inter / max(min(area(a), area(b)), 1e-6)

def union_box(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1,y1,x2,y2)

def frame_idx(name):
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else -1

# Union-Find for clustering overlapping hands
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

# =============== LOAD ===============

def load_rows():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")
    with IN_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows

# =============== MAIN ===============

def main():
    rows = load_rows()

    # group detections by frame
    frames = defaultdict(lambda: {"meta": None, "hands": [], "others": []})
    for r in rows:
        frame = Path(r.get("frame") or "").name
        if not frame:
            continue
        try:
            W = int(float(r.get("img_w","0"))); H = int(float(r.get("img_h","0")))
            x1 = float(r["x1"]); y1 = float(r["y1"]); x2 = float(r["x2"]); y2 = float(r["y2"])
        except Exception:
            continue

        if frames[frame]["meta"] is None:
            frames[frame]["meta"] = {"frame": frame, "img_w": W, "img_h": H}

        if is_hand_row(r):
            frames[frame]["hands"].append({
                "side": (r.get("hand_side") or "").strip().lower(),  # "left", "right", "", "unknown"
                "cls":  get_cls(r),
                "box":  (x1,y1,x2,y2),
                "conf": float(r.get("conf","0") or 0.0)
            })
        else:
            frames[frame]["others"].append({
                "cls": get_cls(r),
                "box": (x1,y1,x2,y2),
                "conf": float(r.get("conf","0") or 0.0)
            })

    # Output rows
    out_labels = []      # per merged/single hand box
    out_summary = []     # per frame label: "separate" vs "together"

    # Iterate frames
    for frame, data in sorted(frames.items(), key=lambda kv: frame_idx(kv[0])):
        meta = data["meta"]
        H = meta["img_h"]; W = meta["img_w"]
        hands = data["hands"]

        # Build image
        img_path = SRC_DIR / frame
        img = cv2.imread(str(img_path)) if img_path.exists() else None

        # No hands → annotate nothing, mark separate
        if not hands:
            out_summary.append({"frame": frame, "state": "no_hands", "num_components": 0, "num_hands": 0})
            if img is not None:
                cv2.imwrite(str(OUT_IMGS / frame), img)
            continue

        # Cluster hands by overlap
        n = len(hands)
        dsu = DSU(n)
        for i in range(n):
            for j in range(i+1, n):
                b1 = hands[i]["box"]; b2 = hands[j]["box"]
                if iou(b1,b2) >= IOU_THRESH or overlap_over_min(b1,b2) >= INTER_OVER_MINAREA_THRESH:
                    dsu.union(i,j)

        # Components
        groups = defaultdict(list)
        for i in range(n):
            groups[dsu.find(i)].append(i)

        # Decide per-component label and box
        components = []
        for root, idxs in groups.items():
            member_boxes = [hands[k]["box"] for k in idxs]
            merged = union_box(member_boxes) if len(idxs) >= 2 else hands[idxs[0]]["box"]
            members_sides = [hands[k]["side"] or "unknown" for k in idxs]
            members_confs = [hands[k]["conf"] for k in idxs]

            if len(idxs) >= 2:
                label = "merged_hands"
            else:
                s = members_sides[0]
                if s in ("left","right"):
                    label = f"{s}_hand"
                else:
                    label = "unknown_hand"

            components.append({
                "label": label,
                "box": merged,
                "members": members_sides,
                "members_n": len(idxs),
                "avg_conf": sum(members_confs)/max(1,len(members_confs))
            })

        # Per-frame state
        state = "together" if any(c["members_n"] >= 2 for c in components) else "separate"
        out_summary.append({
            "frame": frame,
            "state": state,
            "num_components": len(components),
            "num_hands": len(hands)
        })

        # Draw + write per-component rows
        if img is not None:
            for comp in components:
                x1,y1,x2,y2 = map(int, comp["box"])
                if comp["label"] == "merged_hands":
                    col = COL_MERGED
                    tag = f"{comp['label']} ({comp['members_n']})"
                else:
                    col = COL_SINGLE
                    tag = comp["label"]
                cv2.rectangle(img, (x1,y1), (x2,y2), col, THICK)
                cv2.putText(img, tag, (x1, max(10, y1-6)), FONT, TSCALE, col, TTHICK, cv2.LINE_AA)

            cv2.imwrite(str(OUT_IMGS / frame), img)

        # Emit CSV rows
        for comp in components:
            x1,y1,x2,y2 = comp["box"]
            out_labels.append({
                "frame": frame,
                "img_w": W, "img_h": H,
                "label": comp["label"],         # left_hand / right_hand / unknown_hand / merged_hands
                "x1": round(float(x1),1),
                "y1": round(float(y1),1),
                "x2": round(float(x2),1),
                "y2": round(float(y2),1),
                "members": ";".join(comp["members"]),
                "members_n": comp["members_n"],
                "avg_conf": round(comp["avg_conf"],3)
            })

    # Write outputs
    with OUT_LABELS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame","img_w","img_h","label","x1","y1","x2","y2","members","members_n","avg_conf"
        ])
        w.writeheader()
        w.writerows(out_labels)

    with OUT_SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame","state","num_components","num_hands"])
        w.writeheader()
        w.writerows(out_summary)

    print(f"[OK] Wrote labels → {OUT_LABELS_CSV} ({len(out_labels)} rows)")
    print(f"[OK] Wrote summary → {OUT_SUMMARY_CSV} ({len(out_summary)} frames)")
    print(f"[OK] Annotations → {OUT_IMGS}")

if __name__ == "__main__":
    main()
