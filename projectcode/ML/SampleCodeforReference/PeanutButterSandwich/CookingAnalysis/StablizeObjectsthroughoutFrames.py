## this code will stabilize object identities across frames, the idea is to create a stable ID for the peanut butter jar, knife, bread, and hand##
## terms to know, Vessel of Interest - VOI (this is where food is identified, plate or something where user is eating)
## Extracting object - EO (This is spoon, knife, fork, hand which is used to calculate the amount of food extracted from bottle)
## Object of extraction - OOE (This is the food item which is being extracted from the bottle, like peanut butter, jelly, honey etc)


import os, csv, json, math, shutil
from dataclasses import dataclass, field
from collections import deque, Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# ============== CONFIG ==============
FRAMES_DIR   = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis2/"  # folder with frame_00001.jpg etc
TRACKS_CSV   = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis3/metadata/tracks.csv"   # detector+tracker output per frame
OUT_DIR      = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis3/frame_stablizer"              # new "a3"
DRAW_VIS     = True               # write images with stabilized boxes
OCR_EARLY_N  = 60                 # run OCR within first N frames to confirm PB jar
RE_OCR_EVERY = 60                 # optional: re-OCR every N frames when jar is frontal/large
IMG_EXT      = ".jpg"             # or ".png"
CONF_TEXT_TOKENS = ("peanut", "butter", "pb", "skippy")  # keywords to confirm PB

# Class mapping sets (map your raw labels to stable groups)
JAR_SET   = {"bottle", "jar", "cup", "bowl", "donut"}      # your wobblers
KNIFE_SET = {"knife", "fork", "spoon", "silverware", "table-knife", "utensil"}
BREAD_SET = {"bread", "toast", "sandwich"}
HAND_SET  = {"hand", "person"}  # optional; keep if available

# Stabilization params
K_WINDOW    = 7       # rolling window majority size
M_CONSEC    = 3       # hysteresis: need M consecutive for group change (when not yet pinned)
GAP_TOL     = 15      # frames to consider for ID recovery if tracker blips
IOU_THRESH  = 0.5     # IoU for merging track break
USE_REID    = False   # set True if your CSV has reid features (cosine sim)
REID_THRESH = 0.7

# OCR (EasyOCR) - optional but recommended
USE_OCR     = True
OCR_LANGS   = ['en']
OCR_MIN_W   = 80      # only OCR jar crops at least this many pixels wide

# Drawing
COLORS = {
    "pb_jar":  (40, 180, 255),
    "knife":   (0, 255, 0),
    "bread":   (255, 180, 40),
    "hand":    (210, 210, 210),
    "other":   (180, 120, 255),
}
# ====================================

try:
    import easyocr
    READER = easyocr.Reader(OCR_LANGS, gpu=False) if USE_OCR else None
except Exception:
    READER = None
    if USE_OCR:
        print("[WARN] EasyOCR not available. Continuing without OCR confirmation.")


@dataclass
class TrackState:
    canonical: Optional[str] = None     # "pb_jar"/"knife"/"bread"/"hand"/"other"
    pinned: bool = False
    last_k_groups: deque = field(default_factory=lambda: deque(maxlen=K_WINDOW))
    class_counts: Counter = field(default_factory=Counter)
    last_seen: int = -1
    last_bbox: Optional[Tuple[int,int,int,int]] = None
    first_seen: int = -1
    consec_group: Optional[str] = None
    consec_count: int = 0
    ocr_brand: Optional[str] = None
    ocr_conf: float = 0.0
    is_peanut_butter: bool = False


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1.0, (area_a + area_b - inter))


def map_to_group(raw_label: str) -> str:
    rl = raw_label.lower()
    if rl in JAR_SET:   return "jar"
    if rl in KNIFE_SET: return "knife"
    if rl in BREAD_SET: return "bread"
    if rl in HAND_SET:  return "hand"
    return "other"


def draw_box(img, box, label, color):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(15, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def ocr_jar_crop(image, box) -> Tuple[str, float]:
    if READER is None: return ("", 0.0)
    x1,y1,x2,y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0,y1)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0: return ("", 0.0)
    if (x2-x1) < OCR_MIN_W: return ("", 0.0)
    try:
        results = READER.readtext(crop)
        # Concatenate text with confidences
        best_text, best_conf = "", 0.0
        for (_, text, conf) in results:
            if conf > best_conf:
                best_text, best_conf = text, conf
        return (best_text.lower(), float(best_conf))
    except Exception:
        return ("", 0.0)


def load_tracks_csv(csv_path) -> Dict[int, List[dict]]:
    """Return dict: frame_index -> list of det dicts"""
    per_frame = defaultdict(list)
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fr = int(row["frame"])
            det = {
                "track_id": int(row["track_id"]),
                "bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                "raw_label": row.get("raw_label","other"),
                "score": float(row.get("score", 0.0)),
            }
            per_frame[fr].append(det)
    return per_frame


def main():
    ensure_dir(OUT_DIR)
    vis_dir = os.path.join(OUT_DIR, "images")
    ensure_dir(vis_dir)
    anno_path = os.path.join(OUT_DIR, "stabilized_annotations.jsonl")

    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(IMG_EXT)], key=lambda x: int(''.join([c for c in x if c.isdigit()]) or 0))
    tracks_by_frame = load_tracks_csv(TRACKS_CSV)

    track_states: Dict[int, TrackState] = {}
    recently_ended: Dict[int, Tuple[int, Tuple[int,int,int,int]]] = {}  # track_id -> (last_seen_frame, last_bbox)

    # To keep a single canonical PB_JAR entity, even if ID changes
    active_pbjar_track_ids = set()

    with open(anno_path, "w") as anno_out:
        for idx, fname in enumerate(frames, start=1):
            fnum = idx
            img_path = os.path.join(FRAMES_DIR, fname)
            img = cv2.imread(img_path)
            h, w = (img.shape[0], img.shape[1]) if img is not None else (0,0)

            dets = tracks_by_frame.get(fnum, [])

            # Update/initialize track states
            for det in dets:
                tid = det["track_id"]
                bbox = det["bbox"]
                raw_label = det["raw_label"]

                if tid not in track_states:
                    track_states[tid] = TrackState(first_seen=fnum)
                st = track_states[tid]
                st.last_seen = fnum
                st.last_bbox = bbox

                group = map_to_group(raw_label)
                st.last_k_groups.append(group)
                st.class_counts[group] += 1

                # Pinning rules (once jar/knife/bread seen, fix it)
                if not st.pinned:
                    if "jar" in st.class_counts:
                        st.canonical, st.pinned = "jar", True
                    elif "knife" in st.class_counts:
                        st.canonical, st.pinned = "knife", True
                    elif "bread" in st.class_counts:
                        st.canonical, st.pinned = "bread", True
                    elif "hand" in st.class_counts:
                        st.canonical, st.pinned = "hand", True

                # If still not pinned, apply hysteresis on window majority
                if not st.pinned:
                    # majority over last_k_groups
                    counts = Counter(st.last_k_groups)
                    maj_group, _ = counts.most_common(1)[0]
                    if st.consec_group != maj_group:
                        st.consec_group = maj_group
                        st.consec_count = 1
                    else:
                        st.consec_count += 1
                    if st.canonical is None:
                        st.canonical = maj_group
                    elif st.consec_count >= M_CONSEC and st.canonical != maj_group:
                        st.canonical = maj_group

            # Try OCR early to confirm PB_JAR
            if USE_OCR and READER is not None and fnum <= OCR_EARLY_N:
                # pick the largest "jar" candidate this frame
                jar_cands = [(d, (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])) for d in dets if map_to_group(d["raw_label"])=="jar"]
                if jar_cands and img is not None:
                    jar_cands.sort(key=lambda x: -x[1])
                    det = jar_cands[0][0]
                    tid = det["track_id"]
                    st = track_states[tid]
                    text, conf = ocr_jar_crop(img, det["bbox"])
                    if any(tok in text for tok in CONF_TEXT_TOKENS):
                        st.is_peanut_butter = True
                        st.ocr_brand = text
                        st.ocr_conf = conf
                        st.canonical = "jar"
                        st.pinned = True
                        active_pbjar_track_ids.add(tid)

            # If we already have a PB_JAR identified, force any of its detections to pb_jar label
            # And if its ID changed, pick the best candidate to keep PB_JAR continuity
            # Simple continuity: whichever "jar group" track is closest to the last known pb_jar bbox
            if active_pbjar_track_ids:
                # keep only those still present; if none present, try to reassign later
                present = [d for d in dets if d["track_id"] in active_pbjar_track_ids]
                if not present:
                    # try to find a new jar-like candidate by IoU with previous pb_jar bbox
                    last_pb_tid = list(active_pbjar_track_ids)[-1]
                    last_pb = track_states.get(last_pb_tid)
                    if last_pb and last_pb.last_bbox is not None:
                        best_tid, best_iou = None, 0.0
                        for d in dets:
                            g = map_to_group(d["raw_label"])
                            if g != "jar": continue
                            iou = iou_xyxy(last_pb.last_bbox, d["bbox"])
                            if iou > best_iou:
                                best_iou, best_tid = iou, d["track_id"]
                        if best_tid is not None and best_iou >= IOU_THRESH:
                            # transfer identity
                            active_pbjar_track_ids = {best_tid}
                            track_states[best_tid].canonical = "jar"
                            track_states[best_tid].pinned = True
                else:
                    # still present; ensure canonical stays jar
                    for d in present:
                        st = track_states[d["track_id"]]
                        st.canonical = "jar"
                        st.pinned = True

            # Build stabilized outputs for this frame
            out_objs = []
            for d in dets:
                tid = d["track_id"]
                st = track_states[tid]
                stable_group = st.canonical or map_to_group(d["raw_label"])

                # Promote jar to pb_jar if OCR confirmed on this track or we already have an active pb jar
                label_out = stable_group
                if stable_group == "jar":
                    if st.is_peanut_butter or (tid in active_pbjar_track_ids):
                        label_out = "pb_jar"

                out_obj = {
                    "frame": fnum,
                    "track_id": tid,
                    "bbox": [round(float(v),2) for v in d["bbox"]],
                    "stable_label": label_out,
                    "raw_label": d["raw_label"],
                    "score": round(float(d["score"]),3),
                    "is_pb": bool(st.is_peanut_butter),
                    "ocr_brand": st.ocr_brand,
                    "ocr_conf": round(st.ocr_conf,3) if st.ocr_conf else 0.0,
                }
                out_objs.append(out_obj)

                # Draw
                if DRAW_VIS and img is not None:
                    col = COLORS.get(label_out, (200,200,200))
                    draw_box(img, d["bbox"], f"{label_out}#{tid}", col)

            # Save visualization image
            if DRAW_VIS and img is not None:
                cv2.imwrite(os.path.join(vis_dir, fname), img)

            # Write JSONL line
            anno_out.write(json.dumps({"frame": fnum, "objects": out_objs}) + "\n")

    print(f"[OK] Stabilized annotations -> {anno_path}")
    if DRAW_VIS:
        print(f"[OK] Visualized images -> {os.path.join(OUT_DIR, 'images')}")
    print("[TIP] Next: derive mouth_ROI per frame from pb_jar bbox (top 30% + pad) for dunk logic.")

if __name__ == "__main__":
    main()

import supervision

