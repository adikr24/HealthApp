## Detect ALL objects with Detectron2 and save annotated frames + CSV ##

import os
import glob
import csv
from pathlib import Path

import cv2
import numpy as np

# ======= Detectron2 =======
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

# ======= Paths (edit if needed) =======
INPUT_DIR   = "/app/mediaFiles/videos/AnnotateVideos/RawVideos/WaterPour"
OUTPUT_DIR  = "/app/mediaFiles/output/videoOutputs/DetectedFrames/water_volume_output/detectronpass/all_objects"
GLOB_PAT    = "frame_*.jpg"      # your frame naming pattern
CONF_THRESH = 0.35               # confidence threshold for ALL classes

# Choose a Detectron2 model zoo config (COCO)
# Some solid options:
#   "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"      # boxes only (default below)
#   "COCO-Detection/retinanet_R_50_FPN_3x.yaml"        # one-stage detector
#   "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"  # boxes + masks
MODEL_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# Save clean vs. annotated outputs separately
WRITE_ANNOTATIONS = True
ANN_DIR   = os.path.join(OUTPUT_DIR, "annotated_frames")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANN_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

LINE_THICK  = 2
FONT_SCALE  = 0.5
FONT_THICK  = 1

def color_for_class(cls_id: int):
    """Deterministic BGR color per class id (no global RNG state)."""
    # Simple hash -> BGR
    return ( (37 * cls_id) % 255, (17 * cls_id) % 255, (29 * cls_id) % 255 )

def draw_label(img, text, x1, y1, color=(0,255,0)):
    """Draw text with filled background above the box."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    pad = 2
    y1 = max(th + 2*pad, y1)
    cv2.rectangle(img, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), color, -1)
    cv2.putText(img, text, (x1 + pad, y1 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK, cv2.LINE_AA)

def build_predictor(conf_thresh=0.35):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CFG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CFG)
    # Threshold fields differ by head; set both safely:
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thresh
    if hasattr(cfg.MODEL, "RETINANET"):
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh
    cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    return DefaultPredictor(cfg), cfg

def main():
    predictor, cfg = build_predictor(CONF_THRESH)

    # Class name mapping (COCO 80 classes)
    meta_name = cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) > 0 else "coco_2017_train"
    metadata = MetadataCatalog.get(meta_name)
    thing_classes = list(getattr(metadata, "thing_classes", []))
    id_to_name = {i: n for i, n in enumerate(thing_classes)} if thing_classes else {}

    # Collect frames
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, GLOB_PAT)))
    if not frames:
        print(f"No frames found under {INPUT_DIR}/{GLOB_PAT}")
        return

    # CSV path
    csv_path = Path(OUTPUT_DIR) / "detections_all_objects_detectron.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "filename", "class_id", "class_name", "confidence",
            "x1", "y1", "x2", "y2"
        ])

        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                print(f"Skip unreadable: {fp}")
                continue
            H, W = img.shape[:2]
            clean = img.copy()
            ann   = img.copy()

            # --- Detectron2 inference ---
            outputs = predictor(img)
            instances = outputs.get("instances", None)
            det_count = 0

            if instances is not None and len(instances) > 0:
                inst_cpu = instances.to("cpu")
                boxes    = inst_cpu.pred_boxes.tensor.numpy() if inst_cpu.has("pred_boxes") else np.zeros((0,4))
                scores   = inst_cpu.scores.numpy() if inst_cpu.has("scores") else np.zeros((0,))
                classes  = inst_cpu.pred_classes.numpy() if inst_cpu.has("pred_classes") else np.zeros((0,), dtype=int)

                for (x1, y1, x2, y2), conf, cls_id in zip(boxes, scores, classes):
                    x1 = int(max(0, min(x1, W-1))); x2 = int(max(0, min(x2, W-1)))
                    y1 = int(max(0, min(y1, H-1))); y2 = int(max(0, min(y2, H-1)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cls_name = str(id_to_name.get(int(cls_id), int(cls_id)))
                    writer.writerow([Path(fp).name, int(cls_id), cls_name, f"{float(conf):.3f}", x1, y1, x2, y2])
                    det_count += 1

                    if WRITE_ANNOTATIONS:
                        color = color_for_class(int(cls_id))
                        cv2.rectangle(ann, (x1, y1), (x2, y2), color, LINE_THICK)
                        draw_label(ann, f"{cls_name} {conf:.2f}", x1, max(0, y1 - 4), color)

            # Optional empty-row breadcrumb
            if det_count == 0:
                writer.writerow([Path(fp).name, "", "", "", "", "", "", ""])

            # Save frames
            cv2.imwrite(os.path.join(CLEAN_DIR, Path(fp).name), clean)
            if WRITE_ANNOTATIONS:
                cv2.imwrite(os.path.join(ANN_DIR, Path(fp).name), ann)

    print(f"Clean frames  → {CLEAN_DIR}")
    if WRITE_ANNOTATIONS:
        print(f"Annotated    → {ANN_DIR}")
    print(f"CSV (Detectron) → {csv_path}")
    print("process completed")

if __name__ == "__main__":
    main()
