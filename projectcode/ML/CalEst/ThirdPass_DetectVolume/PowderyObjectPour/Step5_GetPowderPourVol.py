#!/usr/bin/env python3
# ------------------------------------------------------------
# Simple Inference for fill-level ResNet (empty/half/full, etc.)
# - Runs inference on all images in the specified folder.
# - Outputs per-image predictions to a CSV file.
# - Prints the most frequent (majority) predicted class.
# ------------------------------------------------------------

from pathlib import Path
import csv
import torch
from PIL import Image
from torchvision import transforms, models
from collections import Counter

# ---------- CONFIG ----------
MODEL_PATH  = Path("/app/mediaFiles/output/videoOutputs/ScoopAnalysis/models/filllevel_resnet18.pt")
IMAGES_DIR  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/AnnotatedFrames")
CSV_OUT     = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/InferenceResults_simple.csv")

BATCH_SIZE  = 64
# ----------------------------

def load_model_and_tf(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    img_size    = ckpt.get("img_size", 224)
    mean        = ckpt.get("mean", [0.485, 0.456, 0.406])
    std         = ckpt.get("std",  [0.229, 0.224, 0.225])

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
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {IMAGES_DIR}")

    model, tf, class_names, img_size = load_model_and_tf(MODEL_PATH, device)
    print(f"[INFO] Loaded model with classes: {class_names}  (img_size={img_size})")

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts])
    print(f"[INFO] Found {len(paths)} images in {IMAGES_DIR}")
    if not paths:
        return

    rows = [("filename", "pred_class", *[f"p_{c}" for c in class_names])]
    class_counter = Counter()  # <-- new counter for class tally

    def run_batch(batch_imgs, batch_paths):
        x = torch.stack(batch_imgs).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu()
        pred_idx = prob.argmax(1).tolist()
        for i, pth in enumerate(batch_paths):
            preds = [f"{prob[i, j]:.4f}" for j in range(len(class_names))]
            pred_class = class_names[pred_idx[i]]
            class_counter[pred_class] += 1   # <-- tally predictions
            rows.append((pth.name, pred_class, *preds))
            print(f"{pth.name:40s} -> {pred_class:10s}  "
                  + "  ".join(f"{class_names[j]}={preds[j]}" for j in range(len(class_names))))

    total = 0
    batch_imgs, batch_paths = [], []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping unreadable: {p} ({e})")
            continue
        batch_imgs.append(tf(img))
        batch_paths.append(p)
        if len(batch_imgs) == BATCH_SIZE:
            run_batch(batch_imgs, batch_paths)
            total += len(batch_imgs)
            batch_imgs, batch_paths = [], []

    if batch_imgs:
        run_batch(batch_imgs, batch_paths)
        total += len(batch_imgs)

    # Write CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Print final summary poll
    print("\n----------------------------")
    print(f"[DONE] Wrote predictions for {total} images → {CSV_OUT}")
    if class_counter:
        print("[SUMMARY] Class counts:")
        for cls, cnt in class_counter.most_common():
            print(f"  {cls:10s}: {cnt}")
        top_cls, top_cnt = class_counter.most_common(1)[0]
        print(f"\n[RESULT] Majority class → {top_cls.upper()} ({top_cnt} occurrences)")
    else:
        print("[SUMMARY] No predictions were made.")
    print("----------------------------")


if __name__ == "__main__":
    main()
