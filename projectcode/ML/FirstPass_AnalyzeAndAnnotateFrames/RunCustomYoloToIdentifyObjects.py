#!/usr/bin/env python3
# RunYoloInferenceAllClasses.py
# Purpose: run inference on all frames using your trained YOLOv8 model (no label filtering)

from projectcode.ML.StandardClasses.RunYoloOnVideoOrImages import YoloRelevantAnnotator
from pathlib import Path

#### CONFIG ##############################################################
# Path to your trained YOLO model
WEIGHTS = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloResultIterations/IterationIResults/union93_yolov8n_6402/weights/best.pt"

# Folder with extracted frames from your input video
FRAMES_DIR = Path("/app/mediaFiles/videos/InputVideos/ProteinShake/ExtractFrames")

# Output folder for annotated results
OUT_DIR_YOLO = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo")

# Inference parameters
EVERY_NTH  = 3          # analyze every 3rd frame
CONF_THRES = 0.2       # confidence threshold (≥ 50%)
MAX_FRAMES = 500       # cap total frames processed

#######################################################################
# Ensure output directory exists
OUT_DIR_YOLO.mkdir(parents=True, exist_ok=True)

# Sanity checks
if not Path(WEIGHTS).exists():
    print(f"[WARN] YOLO weights not found: {WEIGHTS}")
if not FRAMES_DIR.exists():
    print(f"[WARN] Frames directory not found: {FRAMES_DIR}")

# Collect frames
all_frames = sorted(FRAMES_DIR.glob("*.jpg"))
sampled_frames = all_frames[::EVERY_NTH][:MAX_FRAMES]

print(f"[INFO] Total frames found: {len(all_frames)}")
print(f"[INFO] Sampled (every {EVERY_NTH}, max {MAX_FRAMES}): {len(sampled_frames)}")
if sampled_frames:
    print(f"[INFO] First sampled frame: {sampled_frames[0]}")

#######################################################################
# Initialize YOLO annotator
annot = YoloRelevantAnnotator(
    model_weights=WEIGHTS,
    conf_threshold=CONF_THRES,
    relevant_labels_csv="/app/mediaFiles/yolo_settings/YoloLabels/CustomIterationLabel_I.csv",
)

#######################################################################
# Run inference
saved_yolo = 0
for i, fp in enumerate(sampled_frames, start=1):
    out_path = OUT_DIR_YOLO / fp.name
    has_rel = annot.process_image_file(fp, out_path)
    status = "✓" if has_rel else "-"
    print(f"[{i:03d}/{len(sampled_frames)}] {fp.name} -> {status}")
    if has_rel:
        saved_yolo += 1

#######################################################################
print(f"\n[YOLO] Frames with detections saved: {saved_yolo}/{len(sampled_frames)}")
print(f"[YOLO] Output directory: {OUT_DIR_YOLO}")
print("[DONE] Inference complete. Review annotated images above.")


