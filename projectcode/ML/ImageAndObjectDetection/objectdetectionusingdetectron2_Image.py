import os
import cv2
import pathlib
from datetime import datetime

# ---------- USER INPUT ----------
my_input_image      = "mediaFiles/images/YoloTrainingImages/yolotrainingImages/images/train/60e7c73f-IMG_9676.jpg"
my_output_base_path = "/app/mediaFiles/images/labeledimages/detectron_labeled_images/milkgallon/labeled_image_I.jpg"
# --------------------------------

# ============== Utilities ==============
def split_base_path(path_str: str):
    """
    If the user gave a file path with an extension, use its parent as base dir
    and the file stem as base name. If it's a directory, create a default name.
    """
    p = pathlib.Path(path_str)
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        base_dir = p.parent
        base_name = p.stem
    else:
        base_dir = p
        base_name = "labeled_image"
    return str(base_dir), base_name

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare output folders/names
out_base_dir, out_base_name = split_base_path(my_output_base_path)
detectron_dir = ensure_dir(os.path.join(out_base_dir.replace("yolo_labeled_images","detectron_labeled_images")))
yolo_dir      = ensure_dir(os.path.join(out_base_dir.replace("detectron_labeled_images","yolo_labeled_images")))

# ============== Detectron2 ==============
def run_detectron(input_image_path: str):
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog

    # COCO Faster R-CNN R50-FPN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.35
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    predictor = DefaultPredictor(cfg)

    img = cv2.imread(input_image_path)
    outputs = predictor(img)

    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1.0,
        instance_mode=ColorMode.IMAGE
    )
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    labeled_bgr = vis.get_image()[:, :, ::-1]
    return labeled_bgr

# ============== YOLO (Ultralytics) ==============
def run_yolo(input_image_path: str):
    from ultralytics import YOLO
    # COCO-pretrained lightweight model; change to your fine-tuned weights if needed
    model = YOLO("yolov8n.pt")
    res = model.predict(input_image_path, conf=0.35, verbose=False)[0]
    # Ultralytics provides a nice plotted numpy image in BGR
    plotted = res.plot()  # already BGR
    return plotted

# ============== Main ==============
if __name__ == "__main__":
    assert os.path.isfile(my_input_image), f"Input image not found: {my_input_image}"

    # Detectron
    try:
        det_img = run_detectron(my_input_image)
        det_out_path = os.path.join(
            detectron_dir,
            f"{out_base_name}_detectron_{timestamp()}.jpg"
        )
        cv2.imwrite(det_out_path, det_img)
        print(f"[Detectron] Saved: {det_out_path}")
    except Exception as e:
        print(f"[Detectron] ERROR: {e}")

    # YOLO
    try:
        yolo_img = run_yolo(my_input_image)
        yolo_out_path = os.path.join(
            yolo_dir,
            f"{out_base_name}_yolo_{timestamp()}.jpg"
        )
        cv2.imwrite(yolo_out_path, yolo_img)
        print(f"[YOLO] Saved: {yolo_out_path}")
    except Exception as e:
        print(f"[YOLO] ERROR: {e}")
