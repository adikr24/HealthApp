#!/usr/bin/env python3
# merge_and_find_segments.py  (merged labels only)

from pathlib import Path
import csv
from collections import defaultdict

# --------- INPUTS ---------
DETRON_CSV = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/FusedPreview/all_images_summary.csv")
YOLO_CSV   = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo/custom_yolo_detections.csv")

# --------- OUTPUT DIR ---------
META_DIR   = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/Metadata")
META_DIR.mkdir(parents=True, exist_ok=True)
MERGED_OUT = META_DIR / "merged_labels.csv"

# --------- HELPERS ---------
def parse_float(s, default=None):
    try:
        return float(s)
    except:
        return default

def frame_key(fr):
    # numeric sort where possible
    try:
        return (0, int(fr))
    except:
        stem = Path(fr).stem
        digits = ''.join(ch if ch.isdigit() else ' ' for ch in stem).split()
        if digits:
            try:
                return (1, int(digits[-1]))
            except:
                pass
        return (2, fr)

def read_custom_yolo(csv_path):
    frames_seen=set(); frames=[]
    by_frame=defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            fr=r["frame"]
            if fr not in frames_seen:
                frames_seen.add(fr); frames.append(fr)
            x1=parse_float(r.get("x1")); y1=parse_float(r.get("y1"))
            x2=parse_float(r.get("x2")); y2=parse_float(r.get("y2"))
            if None in (x1,y1,x2,y2): 
                continue
            conf=parse_float(r.get("conf","0"), 0.0)
            by_frame[fr].append({
                "source":"custom_yolo",
                "cls_id": r.get("cls_id",""),
                "cls_name": r.get("cls_name",""),
                "conf": conf,
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "img_w": r.get("img_w",""),
                "img_h": r.get("img_h",""),
                "cx_norm": r.get("cx_norm",""),
                "cy_norm": r.get("cy_norm",""),
                "w_norm":  r.get("w_norm",""),
                "h_norm":  r.get("h_norm",""),
            })
    frames.sort(key=frame_key)
    return frames, by_frame

def read_detectron(csv_path):
    by_frame=defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        rdr=csv.DictReader(f)
        for r in rdr:
            fr=r["frame"]
            x1=parse_float(r.get("x1")); y1=parse_float(r.get("y1"))
            x2=parse_float(r.get("x2")); y2=parse_float(r.get("y2"))
            if None in (x1,y1,x2,y2):
                continue
            conf=parse_float(r.get("conf","0"), 0.0)
            by_frame[fr].append({
                "source": (r.get("source") or "detectron_person"),
                "cls_id": r.get("cls_id",""),
                "cls_name": (r.get("cls_name") or "person"),
                "conf": conf,
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "img_w": r.get("img_w",""),
                "img_h": r.get("img_h",""),
                "cx_norm": r.get("cx_norm",""),
                "cy_norm": r.get("cy_norm",""),
                "w_norm":  r.get("w_norm",""),
                "h_norm":  r.get("h_norm",""),
            })
    return by_frame

# --------- LOAD ---------
frames_order, yolo_by_frame = read_custom_yolo(YOLO_CSV)
det_by_frame = read_detectron(DETRON_CSV)

# --------- WRITE MERGED ONLY ---------
header = ("frame","img_w","img_h","source","cls_id","cls_name","conf",
          "x1","y1","x2","y2","cx_norm","cy_norm","w_norm","h_norm")

with open(MERGED_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    for fr in frames_order:
        # YOLO detections
        for d in yolo_by_frame.get(fr, []):
            w.writerow((
                fr, d["img_w"], d["img_h"], d["source"], d["cls_id"], d["cls_name"], f'{d["conf"]:.3f}',
                f'{d["x1"]:.1f}', f'{d["y1"]:.1f}', f'{d["x2"]:.1f}', f'{d["y2"]:.1f}',
                d["cx_norm"], d["cy_norm"], d["w_norm"], d["h_norm"]
            ))
        # Detectron persons (if any)
        for d in det_by_frame.get(fr, []):
            w.writerow((
                fr, d["img_w"], d["img_h"], "detectron_person", d["cls_id"], d["cls_name"], f'{d["conf"]:.3f}',
                f'{d["x1"]:.1f}', f'{d["y1"]:.1f}', f'{d["x2"]:.1f}', f'{d["y2"]:.1f}',
                d["cx_norm"], d["cy_norm"], d["w_norm"], d["h_norm"]
            ))

print(f"[DONE] merged -> {MERGED_OUT}")
