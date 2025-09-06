from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# Load Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Video path
output_file = "/app/mediaFiles/dataset_root/pouringmilk/pouringmilk.mp4"
video = cv2.VideoCapture(output_file)

# Output frames directory
save_dir = "/app/mediaFiles/images/labeledimages"
os.makedirs(save_dir, exist_ok=True)

if not video.isOpened():
    print(f"Error opening video file {output_file}")
else:
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = int(fps * 3)  # last 3 seconds
    start_frame = 0 # max(0, total_frames - frames_to_process)

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Processing last {frames_to_process} frames starting from frame {start_frame}")

    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    identified_objects_summary = {}
    saved_frames = 0
    max_saved_frames = 10

    class_names = MetadataCatalog.get("coco_2017_train").thing_classes

    while saved_frames < max_saved_frames:
        ret, frame = video.read()
        if not ret:
            break

        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes
        pred_scores = instances.scores

        new_object_detected = False

        for i in range(len(pred_classes)):
            if pred_scores[i] >= cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
                object_name = class_names[pred_classes[i].item()]
                prev_count = identified_objects_summary.get(object_name, 0)
                identified_objects_summary[object_name] = prev_count + 1

                # Only save if this object wasn't detected before
                if prev_count == 0:
                    new_object_detected = True

        # Save frame with labels if new object appeared
        if new_object_detected:
            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get("coco_2017_train"), scale=1.0)
            out = v.draw_instance_predictions(instances)
            save_path = os.path.join(save_dir, f"frame_{saved_frames+1}.jpg")
            cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
            print(f"Saved frame {saved_frames+1} with detections: {save_path}")
            saved_frames += 1

    video.release()

    print("\n--- Identified Objects in the Last 3 Seconds ---")
    for obj, count in identified_objects_summary.items():
        print(f"- {obj} (detected {count} times)")
