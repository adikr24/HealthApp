# ---- Detectron -> detections.csv (first 10 frames) ----
import os, glob, csv, cv2, torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

FRAMES_DIR     = "/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II"
DETECTIONS_CSV = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis2/detections.csv"

MODEL_CFG    = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
SCORE_THRESH = 0.5
WEIGHTS_PATH = None  # set to a local .pkl if you’re offline
RELEVANT_SET = {"bottle","cup","bowl","knife","fork","spoon","sandwich","bread","person","donut","jar","toast"}

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in ''.join(ch if ch.isalnum() else ' ' for ch in s).split()]

# Build predictor (✅ correct API + device)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_CFG))   # <--- fix: merge_from_file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
cfg.MODEL.WEIGHTS = WEIGHTS_PATH or model_zoo.get_checkpoint_url(MODEL_CFG)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# Class names
try:
    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
except Exception:
    classes = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
        'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
        'scissors','teddy bear','hair drier','toothbrush'
    ]

# Collect frames
frames = sorted(
    glob.glob(os.path.join(FRAMES_DIR, "*.jpg")) + glob.glob(os.path.join(FRAMES_DIR, "*.png")),
    key=_natural_key
)
os.makedirs(os.path.dirname(DETECTIONS_CSV), exist_ok=True)

rows = 0
with open(DETECTIONS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame","x1","y1","x2","y2","raw_label","score"])
    for i, p in enumerate(frames, start=1):
        img = cv2.imread(p)
        if img is None:
            continue

        inst = predictor(img)["instances"].to("cpu")
        if len(inst) == 0:
            print(f"[warn] frame {i}: 0 boxes")
            continue

        boxes  = inst.pred_boxes.tensor.numpy()
        cls_id = inst.pred_classes.numpy()
        scores = inst.scores.numpy()

        for box, cid, sc in zip(boxes, cls_id, scores):
            label = classes[int(cid)] if int(cid) < len(classes) else f"class_{int(cid)}"
            if RELEVANT_SET and label not in RELEVANT_SET:
                continue
            x1, y1, x2, y2 = map(float, box.tolist())
            w.writerow([i, round(x1,2), round(y1,2), round(x2,2), round(y2,2), label, round(float(sc),3)])
            rows += 1

print(f"[OK] Wrote detections CSV: {DETECTIONS_CSV}  (rows: {rows})")


############# STABLIZE THE IMAGES ################################################################################

import csv, os, glob
import numpy as np
from collections import defaultdict
import cv2
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections

DETECTIONS_CSV = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis2/detections.csv"
FRAMES_DIR     = "/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II"

TRACKS_OUT_DIR = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis3/tracker_metadata"
TRACKS_CSV     = os.path.join(TRACKS_OUT_DIR, "tracks.csv")
TRACK_VIZ_DIR  = os.path.join(TRACKS_OUT_DIR, "images_tracked")

os.makedirs(TRACKS_OUT_DIR, exist_ok=True)
os.makedirs(TRACK_VIZ_DIR, exist_ok=True)

# Load detections per frame
_per_frame = defaultdict(list)
with open(DETECTIONS_CSV, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        fr = int(row["frame"])
        _per_frame[fr].append({
            "xyxy": np.array([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float32),
            "label": row["raw_label"],
            "score": float(row["score"]),
        })

# List frames in order
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))

# Init tracker
tracker = ByteTrack()

with open(TRACKS_CSV, "w", newline="") as tf:
    w = csv.writer(tf)
    w.writerow(["frame","track_id","x1","y1","x2","y2","raw_label","score"])

    for idx, frame_path in enumerate(frames, start=1):
        img = cv2.imread(frame_path)
        dets = _per_frame.get(idx, [])
        if dets:
            xyxy = np.stack([d["xyxy"] for d in dets], axis=0)
            conf = np.array([d["score"] for d in dets], dtype=np.float32)
            det_index = np.arange(len(dets), dtype=np.int32)  # keep link to labels
            sv_det = Detections(xyxy=xyxy, confidence=conf, class_id=det_index)
        else:
            sv_det = Detections.empty()

        # Track
        tracked = tracker.update_with_detections(sv_det)

        # Save rows
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tid = int(tracked.tracker_id[i])
            src_idx = int(tracked.class_id[i])
            label = dets[src_idx]["label"] if dets else "other"
            score = dets[src_idx]["score"] if dets else 0.0
            w.writerow([idx, tid, round(float(x1),2), round(float(y1),2),
                        round(float(x2),2), round(float(y2),2), label, round(float(score),3)])

        # Optional: save visual preview with tracked IDs
        if img is not None and len(tracked) > 0:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
                tid = int(tracked.tracker_id[i])
                label = dets[int(tracked.class_id[i])]["label"] if dets else "obj"
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, f"{label}#{tid}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imwrite(os.path.join(TRACK_VIZ_DIR, os.path.basename(frame_path)), img)

print("[OK] Wrote tracks.csv:", TRACKS_CSV)
print("[OK] Tracked previews saved in:", TRACK_VIZ_DIR)
