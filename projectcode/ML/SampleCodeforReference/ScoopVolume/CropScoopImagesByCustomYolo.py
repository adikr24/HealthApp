#!/usr/bin/env python3
# Scoop cropper: YOLO -> crop scoop -> pad square -> resize 224x224 -> save + CSV manifest

from pathlib import Path
import csv, cv2, math
from ultralytics import YOLO

# ==== PATHS (your request) ====================================================
WEIGHTS    = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloResultIterations/IterationIIResults/union93_yolov8n_6403/weights/best.pt"
FRAMES_DIR = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeII/ExtractFrames/ProteinShake")
OUT_BASE   = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/AnnotatedFrames")
# ==============================================================================

# SCOOP-ONLY
ONLY_CLASS_NAME = "scoop"     # change if your YOLO label differs (e.g., "scoop_spoon")

# Inference knobs
EVERY_NTH  = 1
MAX_FRAMES = 2000
CONF_THRES = 0.5
IOU_THRES  = 0.45

# Crop settings
TARGET_SIZE  = 224
CONTEXT_FRAC = 0.15           # 15% margin around box
INTERP       = cv2.INTER_AREA

# Outputs
OUT_CROPS   = OUT_BASE / "scoop_crops_224" / "unalabeled"
MANIFEST_CSV = OUT_BASE / "scoop_crops_manifest.csv"

# Setup
OUT_CROPS.mkdir(parents=True, exist_ok=True)
if not Path(WEIGHTS).exists():
    print(f"[WARN] weights not found: {WEIGHTS}")
if not FRAMES_DIR.exists():
    print(f"[WARN] frames dir not found: {FRAMES_DIR}")

# Gather frames
all_frames = sorted(list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.png")))
frames = all_frames[::EVERY_NTH][:MAX_FRAMES]
print(f"[INFO] frames: total={len(all_frames)} sampled={len(frames)}")

# YOLO model and class mapping
yolo = YOLO(str(WEIGHTS))
id2name = yolo.model.names if hasattr(yolo.model, "names") else {}
wanted_ids = {cid for cid, cname in id2name.items()
              if str(cname).strip().lower() == ONLY_CLASS_NAME.strip().lower()}
if not wanted_ids:
    print(f"[WARN] class '{ONLY_CLASS_NAME}' not in model names: {id2name}")

def expand_and_square(x1,y1,x2,y2, W,H, margin=CONTEXT_FRAC):
    w = x2 - x1; h = y2 - y1
    cx = x1 + w/2.0; cy = y1 + h/2.0
    w2 = w * (1 + 2*margin); h2 = h * (1 + 2*margin)
    side = max(w2, h2)
    x1s = cx - side/2.0; y1s = cy - side/2.0
    x2s = cx + side/2.0; y2s = cy + side/2.0
    x1s = max(0, math.floor(x1s)); y1s = max(0, math.floor(y1s))
    x2s = min(W, math.ceil(x2s));  y2s = min(H, math.ceil(y2s))
    # ensure square after clamping
    side_clamped = max(x2s - x1s, y2s - y1s)
    x2s = min(W, x1s + side_clamped); y2s = min(H, y1s + side_clamped)
    return int(x1s), int(y1s), int(x2s), int(y2s)

manifest_rows = [("crop_path","frame","img_w","img_h",
                  "cls_id","cls_name","conf",
                  "x1","y1","x2","y2","x1_sq","y1_sq","x2_sq","y2_sq")]

saved = 0
for i, fp in enumerate(frames, 1):
    img = cv2.imread(str(fp))
    if img is None:
        print(f"[WARN] unreadable: {fp}")
        continue
    H, W = img.shape[:2]

    res = yolo.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print(f"[{i:04d}] {fp.name}: no detections")
        continue

    count = 0
    for b in res.boxes:
        cls_id = int(b.cls.item())
        if cls_id not in wanted_ids:
            continue
        conf = float(b.conf.item())
        x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())

        x1s,y1s,x2s,y2s = expand_and_square(x1,y1,x2,y2, W,H)
        crop = img[y1s:y2s, x1s:x2s]
        if crop.size == 0:
            continue

        crop224 = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=INTERP)
        out_name = f"{fp.stem}_scoop_{count:02d}_{TARGET_SIZE}.png"
        out_path = OUT_CROPS / out_name
        cv2.imwrite(str(out_path), crop224)
        saved += 1; count += 1

        manifest_rows.append((
            str(out_path), fp.name, W, H,
            cls_id, id2name.get(cls_id, f"cls_{cls_id}"), f"{conf:.3f}",
            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
            x1s, y1s, x2s, y2s
        ))

    print(f"[{i:04d}/{len(frames)}] {fp.name}: saved {count} scoop crop(s)")

# Write manifest
MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST_CSV, "w", newline="") as f:
    csv.writer(f).writerows(manifest_rows)

print(f"\n[DONE] Saved {saved} crops → {OUT_CROPS}")
print(f"[DONE] Manifest CSV → {MANIFEST_CSV}")
print("Next: move crops into 000/025/050/075/100 folders (train/val) for ResNet.")
