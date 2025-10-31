#!/usr/bin/env python3
# Detect only the VOI (e.g. blender) for frame ranges listed in vol_detection_reference.csv
# Draws bbox only for that VOI and saves annotated frames.

from pathlib import Path
import csv, cv2
from ultralytics import YOLO

#### CONFIG ##############################################################
WEIGHTS = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloResultIterations/IterationIIResults/union93_yolov8n_6403/weights/best.pt"
FRAMES_DIR = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeII/ExtractFrames/ProteinShake")
VOL_REF = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/FSM/fsm_heuristics_Merged.csv")
OUT_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/VOIYolo/")

CONF_THRES = 0.5
IOU_THRES = 0.45
#######################################################################


def detect_voi_in_ranges(vol_ref_csv: Path, frames_dir: Path, out_dir: Path, weights: Path):
    """Detect only the VOI for each frame range defined in vol_ref_csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(weights))
    id2name = yolo.model.names if hasattr(yolo.model, "names") else {}

    def load_frame(idx: int):
        for ext in [".jpg", ".png"]:
            fp = frames_dir / f"frame_{idx:05d}{ext}"
            if fp.exists():
                return fp
        return None

    # read CSV
    ranges = []
    with open(vol_ref_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                start = int(r["start_idx"])
                end = int(r["end_idx"])
                voi = r["VOI"].strip().lower()
                ranges.append((start, end, voi))
            except:
                continue

    # run detection
    for (start, end, voi_name) in ranges:
        for idx in range(start, end + 1):
            fp = load_frame(idx)
            if fp is None:
                continue
            print(f"[INFO] Processing {fp.name}")
            img = cv2.imread(str(fp))
            if img is None:
                continue

            res = yolo.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
            if not res.boxes:
                continue

            for b in res.boxes:
                cls_id = int(b.cls.item())
                cls_name = id2name.get(cls_id, "").lower()
                if voi_name not in cls_name:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf.item())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imwrite(str(out_dir / fp.name), img)


if __name__ == "__main__":
    detect_voi_in_ranges(VOL_REF, FRAMES_DIR, OUT_DIR, WEIGHTS)
