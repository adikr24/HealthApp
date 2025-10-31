#!/usr/bin/env python3
# ------------------------------------------------------------
# Inference for fill-level ResNet (empty/half/full, etc.)
# - No CLI args needed. Just run in VS Code terminal.
# - Loads checkpoint, scans a folder of images, prints results,
#   and writes a CSV alongside.
# ------------------------------------------------------------

from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms, models

# ---------- CONFIG ----------
MODEL_PATH  = Path("/app/mediaFiles/output/videoOutputs/ScoopAnalysis/models/filllevel_resnet18.pt")
IMAGES_DIR  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/AnnotatedFrames")
CSV_OUT     = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/InferenceResults")
BATCH_SIZE  = 64
# ----------------------------

def load_model_and_tf(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    img_size    = ckpt.get("img_size", 224)
    mean        = ckpt.get("mean", [0.485, 0.456, 0.406])
    std         = ckpt.get("std",  [0.229, 0.224, 0.225])

    # Build model skeleton and load weights
    m = models.resnet18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(class_names))
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return m, tf, class_names, img_size

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    model, tf, class_names, img_size = load_model_and_tf(MODEL_PATH, device)
    print(f"[INFO] Loaded model with classes: {class_names}  (img_size={img_size})")

    # Collect images in folder (non-recursive)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts])
    print(f"[INFO] Found {len(paths)} images in: {IMAGES_DIR}")
    if not paths:
        print("[WARN] No images found. Check IMAGES_DIR.")
        return

    # Batched inference
    rows = [("filename", "pred_class", *[f"p_{c}" for c in class_names])]
    batch, batch_paths = [], []
    total = 0

    def run_batch(batch_imgs, batch_ps):
        nonlocal rows
        x = torch.stack(batch_imgs).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu()
        pred_idx = prob.argmax(1).tolist()
        for i, pth in enumerate(batch_ps):
            preds = [f"{prob[i, j]:.4f}" for j in range(len(class_names))]
            print(f"{pth.name:40s} -> {class_names[pred_idx[i]]:10s}  "
                  + "  ".join(f"{class_names[j]}={preds[j]}" for j in range(len(class_names))))
            rows.append((pth.name, class_names[pred_idx[i]], *preds))

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping unreadable: {p} ({e})")
            continue
        batch.append(tf(img))
        batch_paths.append(p)
        if len(batch) == BATCH_SIZE:
            run_batch(batch, batch_paths)
            total += len(batch)
            batch, batch_paths = [], []

    if batch:
        run_batch(batch, batch_paths)
        total += len(batch)

    # Write CSV
    with open(CSV_OUT, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[DONE] Wrote predictions for {total} images → {CSV_OUT}")

if __name__ == "__main__":
    main()
