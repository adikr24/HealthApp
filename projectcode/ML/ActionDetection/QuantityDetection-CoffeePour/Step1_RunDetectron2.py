# detectron2_spoon_in_cup_windows.py
from detectron2.utils.logger import setup_logger
setup_logger()

import os, glob
from pathlib import Path
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# --------- INPUTS ---------
INPUT_DIR  = "/app/mediaFiles/videos/AnnotateVideos/RawVideos/CoffeePour/CoffeePour"
GLOB_PAT   = "frame_*.jpg"              # frames like frame_00000.jpg
FIRST_N    = None                       # set to an int (e.g., 300) or None for ALL
SCORE_THR  = 0.7                        # detection confidence threshold

# --------- SAVE ONLY WINDOWS AROUND SPOON-IN-CUP ---------
BEFORE_FRAMES = 12                      # save this many frames before
AFTER_FRAMES  = 12                      # ...and this many after (set 10/15 as you like)

# How to decide if spoon is "in" cup
INSIDE_MODE   = "center-in"             # "center-in" or "iou"
IOU_MIN       = 0.10                     # used if INSIDE_MODE == "iou"
CUP_NAME      = "cup"                   # COCO has 'cup'
SPOON_NAME    = "spoon"                 # COCO has 'spoon'

# --------- OUTPUTS ---------
OUT_DIR    = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/detectron2_frames"
os.makedirs(OUT_DIR, exist_ok=True)

def build_predictor(score_thr=0.7):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    # Optional: force CPU
    # cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg), cfg

def bbox_center(b):
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def center_in(b_spoon, b_cup):
    cx, cy = bbox_center(b_spoon)
    x1, y1, x2, y2 = b_cup
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union

def spoon_in_cup(spoon_boxes, cup_boxes):
    if len(spoon_boxes) == 0 or len(cup_boxes) == 0:
        return False
    for s in spoon_boxes:
        for c in cup_boxes:
            if INSIDE_MODE == "center-in":
                if center_in(s, c):
                    return True
            else:  # "iou"
                if iou(s, c) >= IOU_MIN:
                    return True
    return False

def main():
    # Collect frames
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, GLOB_PAT)))
    if not frames:
        print(f"No frames found under {INPUT_DIR}/{GLOB_PAT}")
        return
    if FIRST_N is not None:
        frames = frames[:FIRST_N]

    predictor, cfg = build_predictor(SCORE_THR)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Class indices
    class_names = metadata.thing_classes
    try:
        cup_idx   = class_names.index(CUP_NAME)
        spoon_idx = class_names.index(SPOON_NAME)
    except ValueError:
        print(f"ERROR: Required classes not found in metadata. Available: {class_names}")
        return

    print(f"Scanning {len(frames)} frame(s) from: {INPUT_DIR}")
    trig = [False] * len(frames)

    # ---- PASS 1: find frames where spoon ∈ cup ----
    for i, fp in enumerate(frames):
        img = cv2.imread(fp)
        if img is None:
            continue
        outputs = predictor(img)
        inst = outputs["instances"].to("cpu")

        keep = inst.scores >= SCORE_THR
        if keep.sum() == 0:
            continue

        boxes = inst.pred_boxes.tensor.numpy()
        clses = inst.pred_classes.numpy()

        spoon_boxes = boxes[clses == spoon_idx] if (clses == spoon_idx).any() else []
        cup_boxes   = boxes[clses == cup_idx]   if (clses == cup_idx).any()   else []

        if len(spoon_boxes) and len(cup_boxes):
            if spoon_in_cup(spoon_boxes, cup_boxes):
                trig[i] = True

    # Build set of indices to save (windows around triggers)
    to_save = set()
    for i, is_trig in enumerate(trig):
        if is_trig:
            a = max(0, i - BEFORE_FRAMES)
            b = min(len(frames)-1, i + AFTER_FRAMES)
            for j in range(a, b+1):
                to_save.add(j)

    print(f"Triggered frames: {sum(trig)}")
    print(f"Frames to save (with windows): {len(to_save)}")

    # ---- PASS 2: draw & save ONLY selected frames ----
    for i in sorted(to_save):
        fp = frames[i]
        img = cv2.imread(fp)
        if img is None:
            print(f"Skip unreadable: {fp}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        v = v.draw_instance_predictions(instances)
        annotated = v.get_image()[:, :, ::-1]

        out_path = os.path.join(OUT_DIR, Path(fp).name)
        cv2.imwrite(out_path, annotated)

        n = len(instances) if hasattr(instances, "pred_boxes") else 0
        print(f"[save] {Path(fp).name}: detections={n} -> {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
