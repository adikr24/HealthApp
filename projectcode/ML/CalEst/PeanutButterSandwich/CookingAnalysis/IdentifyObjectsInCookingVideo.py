### Ideally ouput from this would be a collapsed estimation of food object eaten by the preson in the video ###

from projectcode.ML.StandardClasses.RunYoloOnVideoOrImages import YoloRelevantAnnotator
from projectcode.ML.StandardClasses.RunDetectronOnVideosOrImages import DetectronRelevantAnnotator
import numpy as np
from collections import defaultdict
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections
import csv
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import os
import glob
import cv2



#### RUN YOLO FIRST ####################################################################
# Choose best weights now; easy to swap later
WEIGHTS = "/app/mediaFiles/yolo_results/YoloWeights/yolov8n.pt" 
CSV     = "/app/mediaFiles/yolo_results/YoloLabels/YoloRelevantLabels.csv"

#/app/mediaFiles/yolo_results/YoloWeights/yolov8l.pt

annot = YoloRelevantAnnotator(
    model_weights=WEIGHTS,
    conf_threshold=0.5,
    relevant_labels_csv=CSV,
)
# step 1 --> process frames in a folder and save results
## cooking video path ## --> HealthApp\mediaFiles\videos\InputVideos\PeanutButterSandwich\ExtractFrames\MakingPeanutButterSandwich_I
saved_analysis = annot.process_frames_dir(
    frames_dir="/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II",
    output_dir="/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Yolo/Analysis2",
    glob_pat="*.jpg",   # or "frame_*.jpg" if that’s your naming convention
    save_empty=False,
    max_frames=None,  # process all frames
)

print("Frames saved:", saved_analysis)

######## RUN DETECTRON NEXT ##############################################################
det = DetectronRelevantAnnotator(
    model_cfg="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",  # swap to another zoo cfg if you like
    score_threshold=0.5,                                      # tweak like YOLO's conf
    relevant_labels_csv=CSV,                                  # same CSV filter (Relevant == 'Y')
    # weights_path="/path/to/custom_detectron_weights.pth",   # optional custom weights
)

saved_detectron = det.process_frames_dir(
    frames_dir="/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II",
    output_dir="/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis2",
    glob_pat="*.jpg",
    save_empty=False,
    max_frames=None,  # process all frames
)

print("Detectron frames saved:", saved_detectron)

### Assuming sandwich is eaten in all frames ###
# ================== STEP A: EXPORT DETECTRON DETECTIONS → detections.csv ==================
# Re-runs Detectron on the same frames to serialize per-frame boxes into a CSV
# (If your DetectronRelevantAnnotator already writes a CSV, skip this block and just set DETECTIONS_CSV to it.)

FRAMES_DIR = "/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II"
DETECTIONS_CSV = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis2/detections.csv"

MODEL_CFG     = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
SCORE_THRESH  = 0.5
WEIGHTS_PATH  = None  # or your custom .pth if you used one above
RELEVANT_SET  = {"bottle","cup","bowl","knife","fork","spoon","sandwich","bread","person","donut","jar","toast"}

def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in ''.join(ch if ch.isalnum() else ' ' for ch in s).split()]

# Build predictor consistent with your earlier config
_cfg = get_cfg()
_cfg.mergeFromFile(model_zoo.get_config_file(MODEL_CFG))
_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH
_cfg.MODEL.WEIGHTS = WEIGHTS_PATH or model_zoo.get_checkpoint_url(MODEL_CFG)
_predictor = DefaultPredictor(_cfg)

# COCO classes (fallback list if metadata not present)
try:
    _classes = MetadataCatalog.get(_cfg.DATASETS.TRAIN[0]).thing_classes
except Exception:
    _classes = [
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

_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")) + glob.glob(os.path.join(FRAMES_DIR, "*.png")), key=_natural_key)
os.makedirs(os.path.dirname(DETECTIONS_CSV), exist_ok=True)


with open(DETECTIONS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame","x1","y1","x2","y2","raw_label","score"])
    for i, p in enumerate(_frames, start=1):
        img = cv2.imread(p)
        if img is None: 
            continue
        out = _predictor(img)["instances"].to("cpu")
        if len(out) == 0:
            continue
        boxes = out.pred_boxes.tensor.numpy()
        cls_ids = out.pred_classes.numpy()
        scores = out.scores.numpy()
        for box, cid, sc in zip(boxes, cls_ids, scores):
            label = _classes[int(cid)] if int(cid) < len(_classes) else f"class_{int(cid)}"
            if RELEVANT_SET and label not in RELEVANT_SET:
                continue
            x1,y1,x2,y2 = box.tolist()
            w.writerow([i, round(x1,2), round(y1,2), round(x2,2), round(y2,2), label, round(float(sc),3)])

print("[OK] Wrote detections CSV:", DETECTIONS_CSV)

# ================== STEP B: RUN BYTETRACK (supervision) → tracks.csv ==================
# Consumes detections.csv, assigns stable IDs, writes tracks.csv, and optional tracked previews.

TRACKS_OUT_DIR = "/app/mediaFiles/output/videoOutputs/PBJResults/CookingAnalysis/IngredientsIdentification/Detectron/Analysis3/tracker_metadata"
TRACKS_CSV     = os.path.join(TRACKS_OUT_DIR, "tracks.csv")
TRACK_VIZ_DIR  = os.path.join(TRACKS_OUT_DIR, "images_tracked")  # optional viz

os.makedirs(TRACKS_OUT_DIR, exist_ok=True)
os.makedirs(TRACK_VIZ_DIR, exist_ok=True)

# Load detections into frame-wise dict
from collections import defaultdict
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

# List frames again to iterate consistently
_frames_list = sorted([os.path.basename(p) for p in _frames], key=_natural_key)

from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.core import Detections

tracker = ByteTrack()  # no kwargs in this version

with open(TRACKS_CSV, "w", newline="") as tf:
    w = csv.writer(tf)
    w.writerow(["frame","track_id","x1","y1","x2","y2","raw_label","score"])

    for idx, fname in enumerate(_frames_list, start=1):
        fnum = idx
        img_path = os.path.join(FRAMES_DIR, fname)
        img = cv2.imread(img_path)
        dets = _per_frame.get(fnum, [])

        # Build Supervision Detections for this frame
        if dets:
            xyxy = np.stack([d["xyxy"] for d in dets], axis=0)
            conf = np.array([d["score"] for d in dets], dtype=np.float32)
            det_index = np.arange(len(dets), dtype=np.int32)  # index back into dets
            sv_det = Detections(xyxy=xyxy, confidence=conf, class_id=det_index)
        else:
            sv_det = Detections.empty()

        # >>> UPDATED LINE: use update_with_detections (no frame_shape arg)
        tracked = tracker.update_with_detections(detections=sv_det)

        # Write rows to tracks.csv
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            tid             = int(tracked.tracker_id[i])
            src_idx         = int(tracked.class_id[i])  # maps back to dets list
            raw_label       = dets[src_idx]["label"] if dets else "other"
            score           = dets[src_idx]["score"] if dets else 0.0
            w.writerow([
                fnum, tid,
                round(float(x1),2), round(float(y1),2),
                round(float(x2),2), round(float(y2),2),
                raw_label, round(float(score),3)
            ])

        # Optional visualization
        if img is not None and len(tracked) > 0:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = [int(v) for v in tracked.xyxy[i]]
                tid            = int(tracked.tracker_id[i])
                src_idx        = int(tracked.class_id[i])
                lbl            = dets[src_idx]["label"] if dets else "obj"
                col            = _rand_color(tid)
                cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
                cv2.putText(img, f"{lbl}#{tid}", (x1, max(18,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if img is not None:
            cv2.imwrite(os.path.join(TRACK_VIZ_DIR, fname), img)

print("[OK] Wrote tracker output:", TRACKS_CSV)
print("[OK] Tracked previews:", TRACK_VIZ_DIR)


############################################################################################
## 
import glob, os
FRAMES_DIR = "/app/mediaFiles/videos/InputVideos/PeanutButterSandwich/ExtractFrames/MakingPeanutButterSandwich_II"
frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
print("First frame:", frames[0])

out = _predictor(cv2.imread(frames[0]))["instances"].to("cpu")
print("num boxes:", len(out))
if len(out)>0:
    print(out.pred_boxes.tensor.numpy()[:3], out.pred_classes.numpy()[:3], out.scores.numpy()[:3])

