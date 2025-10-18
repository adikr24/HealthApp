#!/usr/bin/env python3
# detectron_person_export_and_visualize.py
# Simple: run Detectron2, draw PERSON boxes, save CSV and optional YOLO .txt

from pathlib import Path
import glob, csv
import cv2
import numpy as np

# ---------------- CONFIG ----------------
FRAMES_DIR   = Path("/app/mediaFiles/videos/InputVideos/ProteinShake/ExtractFrames")
OUT_VIS_DIR  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/DetectronPerson_Vis")
OUT_CSV      = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/DetectronPerson/detections.csv")
OUT_TXT_DIR  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/DetectronPerson/txt")  # YOLO txt per frame
WRITE_TXT    = True         # set False if you only want CSV
PERSON_CLASS_ID_FOR_TXT = 900  # keep distinct from your custom YOLO ids
MAX_FRAMES   = 2000         # None for all frames
SCORE_THRESH = 0.50         # confidence threshold

# -------------- Setup paths -------------
OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
if WRITE_TXT:
    OUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

# -------------- Detectron2 --------------
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def build_predictor(score_th=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_th
    cfg.MODEL.DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    return DefaultPredictor(cfg)

def to_yolo_norm(x1,y1,x2,y2, W,H):
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    return (cx/W, cy/H, w/W, h/H)

def main():
    # collect frames
    frames = sorted(glob.glob(str(FRAMES_DIR / "*.jpg")) + glob.glob(str(FRAMES_DIR / "*.png")))
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    if not frames:
        print(f"[WARN] No frames found in {FRAMES_DIR}")
        return

    predictor = build_predictor(SCORE_THRESH)

    # CSV header
    rows = [("frame","det_idx","x1","y1","x2","y2","score")]

    for i, fp in enumerate(frames, 1):
        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]
        name = Path(fp).name

        out = predictor(img)
        inst = out["instances"]
        if len(inst) > 0:
            boxes  = inst.pred_boxes.tensor.cpu().numpy()
            clss   = inst.pred_classes.cpu().numpy().tolist()
            scores = inst.scores.cpu().numpy().tolist()
        else:
            boxes  = np.zeros((0,4), dtype=np.float32)
            clss   = []
            scores = []

        # filter person only (COCO class 0)
        person_idxs = [k for k,c in enumerate(clss) if c == 0]
        person_boxes  = boxes[person_idxs] if len(person_idxs) else np.zeros((0,4))
        person_scores = [scores[k] for k in person_idxs]

        # draw and (optionally) write per-frame YOLO txt
        if WRITE_TXT:
            with open(OUT_TXT_DIR / f"{Path(name).stem}.txt", "w") as f:
                for (x1,y1,x2,y2), sc in zip(person_boxes, person_scores):
                    cx,cy,w,h = to_yolo_norm(float(x1),float(y1),float(x2),float(y2), W,H)
                    f.write(f"{PERSON_CLASS_ID_FOR_TXT} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}\n")

        # add to CSV rows
        for k, ((x1,y1,x2,y2), sc) in enumerate(zip(person_boxes, person_scores)):
            rows.append((name, k, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{sc:.3f}"))

        # draw boxes on image
        vis = img
        for (x1,y1,x2,y2), sc in zip(person_boxes, person_scores):
            cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.putText(vis, f"person {sc:.2f}", (int(x1), max(0,int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.imwrite(str(OUT_VIS_DIR / name), vis)

        if i % 100 == 0:
            print(f"[INFO] {i}/{len(frames)} frames processed")

    # write CSV
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"[DONE] Visuals: {OUT_VIS_DIR}")
    print(f"[DONE] CSV:     {OUT_CSV}")
    if WRITE_TXT:
        print(f"[DONE] YOLO txt per frame: {OUT_TXT_DIR} (class {PERSON_CLASS_ID_FOR_TXT})")

if __name__ == "__main__":
    main()
