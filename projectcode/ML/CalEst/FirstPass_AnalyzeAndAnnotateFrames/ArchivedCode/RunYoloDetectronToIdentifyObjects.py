from projectcode.ML.StandardClasses.RunYoloOnVideoOrImages import YoloRelevantAnnotator
from projectcode.ML.StandardClasses.RunDetectronOnVideosOrImages import DetectronRelevantAnnotator
from pathlib import Path

#### CONFIG ##############################################################
WEIGHTS = "/app/mediaFiles/yolo_results/YoloWeights/yolov8n.pt"
CSV     = "/app/mediaFiles/yolo_settings/YoloLabels/YoloRelevantLabels.csv"
FRAMES_DIR = Path("/app/mediaFiles/videos/InputVideos/ProteinShake/ExtractFrames")
OUT_DIR_YOLO = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/Yolo")
OUT_DIR_DETECTRON = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/Detectron")

EVERY_NTH  = 3         # analyze every 3rd frame
CONF_THRES = 0.50      # >= 50% confidence
MAX_FRAMES = 600       # <-- back on

# ensure output dirs exist
OUT_DIR_YOLO.mkdir(parents=True, exist_ok=True)
OUT_DIR_DETECTRON.mkdir(parents=True, exist_ok=True)

# quick checks
if not Path(CSV).exists():
    print(f"[WARN] Relevant-labels CSV not found at: {CSV}")
if not FRAMES_DIR.exists():
    print(f"[WARN] Frames dir not found: {FRAMES_DIR}")

#### YOLO ###############################################################
annot = YoloRelevantAnnotator(
    model_weights=WEIGHTS,
    conf_threshold=CONF_THRES,
    relevant_labels_csv=CSV,
)

all_frames = sorted(FRAMES_DIR.glob("*.jpg"))
sampled_frames = all_frames[::EVERY_NTH][:MAX_FRAMES]

print(f"[INFO] Total frames found: {len(all_frames)} | Sampled (every {EVERY_NTH}): {len(sampled_frames)}")
if sampled_frames:
    print(f"[INFO] First sampled frame: {sampled_frames[0]}")

saved_yolo = 0
for fp in sampled_frames:
    has_rel = annot.process_image_file(fp, OUT_DIR_YOLO / fp.name)
    if has_rel:
        saved_yolo += 1
print(f"[YOLO] Frames with relevant detections saved: {saved_yolo}/{len(sampled_frames)}")

#### DETECTRON ##########################################################
det = DetectronRelevantAnnotator(
    model_cfg="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    score_threshold=CONF_THRES,
    relevant_labels_csv=CSV,
)

saved_d2 = 0
for fp in sampled_frames:
    has_rel = det.process_image_file(fp, OUT_DIR_DETECTRON / fp.name)
    if has_rel:
        saved_d2 += 1
print(f"[Detectron] Frames with relevant detections saved: {saved_d2}/{len(sampled_frames)}")
