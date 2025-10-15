#!/usr/bin/env python3
"""
Create YOLO train/val split using ONLY labeled samples.
- Assumes YOLO-style dirs: ROOT/yolo_converted/images and ROOT/yolo_converted/labels
- Keeps samples where the corresponding label file exists AND has at least one non-empty line.
- Splits into train/val with a fixed seed.
- Uses hardlinks to save space; falls back to copy if linking isn't supported.
- Writes a data.yaml pointing at the split.
"""

import os, errno, shutil, random, sys
from pathlib import Path
from typing import List, Tuple

# ========= EDIT THESE =========
ROOT = Path("/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/IterationI")
SRC_DIR = ROOT / "yolo_converted"
IMAGES_DIR = SRC_DIR / "images"
LABELS_DIR = SRC_DIR / "labels"

OUT_DIR = ROOT / "yolo_split"        # will create: train/{images,labels}, val/{images,labels}
VAL_RATIO = 0.10                      # 10% val, 90% train
SEED = 42
# =================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def has_labels(label_path: Path) -> bool:
    """Return True if label file exists and contains at least one non-empty line."""
    if not label_path.exists() or not label_path.is_file():
        return False
    # Fast check: size > 0
    try:
        if label_path.stat().st_size <= 0:
            return False
    except OSError:
        return False
    # Content check (ignore pure whitespace)
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # any non-empty line
                    return True
    except UnicodeDecodeError:
        # If encoding weird, fall back to size-only
        return label_path.stat().st_size > 0
    return False

def place_file(src: Path, dst: Path) -> str:
    """Create dst via hardlink; fallback to copy on cross-device/perm errors."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError as e:
        if e.errno in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            shutil.copy2(src, dst)
            return "copy"
        raise

def collect_labeled_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Return list of (image_path, label_path) where label is non-empty and filenames match by stem."""
    pairs = []
    # Iterate over label files (fewer than images) — keeps only positively labeled samples
    for lbl in labels_dir.glob("*.txt"):
        if not has_labels(lbl):
            continue
        stem = lbl.stem
        # Find matching image (prefer common extensions)
        img = None
        for ext in IMG_EXTS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img = candidate
                break
        if img is None:
            # fallback: scan for any file with same stem
            hits = list(images_dir.glob(stem + ".*"))
            hits = [h for h in hits if h.suffix.lower() in IMG_EXTS]
            if hits:
                img = hits[0]
        if img is not None:
            pairs.append((img, lbl))
    return pairs

def write_data_yaml(out_dir: Path):
    """Write a minimal Ultralytics data.yaml pointing to the split dirs."""
    yaml_path = out_dir / "data.yaml"
    content = f"""# Auto-generated
path: {out_dir.as_posix()}
train: {('train/images')}
val: {('val/images')}
names:
"""
    # read classes.txt if present to include names
    classes_txt = SRC_DIR / "classes.txt"
    names_lines = []
    if classes_txt.exists():
        names_lines = [ln.strip() for ln in classes_txt.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if names_lines:
        # names: [class0, class1, ...] (YAML list)
        content += "  " + str(names_lines) + "\n"
    else:
        content += "  []\n"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path

def main():
    assert IMAGES_DIR.exists(), f"Missing images dir: {IMAGES_DIR}"
    assert LABELS_DIR.exists(), f"Missing labels dir: {LABELS_DIR}"

    pairs = collect_labeled_pairs(IMAGES_DIR, LABELS_DIR)
    total_labeled = len(pairs)
    if total_labeled == 0:
        print("[ERROR] No labeled samples found (non-empty label files).")
        sys.exit(1)

    random.seed(SEED)
    random.shuffle(pairs)

    val_count = max(1, int(total_labeled * VAL_RATIO))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    # Make dirs
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Place files
    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for img, lbl in split_pairs:
            dst_img = OUT_DIR / split / "images" / img.name
            dst_lbl = OUT_DIR / split / "labels" / lbl.name
            place_file(img, dst_img)
            place_file(lbl, dst_lbl)
            stats[split]["images"] += 1
            stats[split]["labels"] += 1

    yaml_path = write_data_yaml(OUT_DIR)

    print("\n=== YOLO Split Complete ===")
    print(f"Total labeled samples: {total_labeled}")
    print(f"Train: {stats['train']['images']}  | Val: {stats['val']['images']}")
    print(f"Output dir: {OUT_DIR}")
    print(f"data.yaml:  {yaml_path}")
    print("\nYou can now train with:")
    print(f"  yolo detect train data={yaml_path.as_posix()} model=yolov8n.pt imgsz=640")
    print("\n(Uses hardlinks when possible; falls back to copy.)")

if __name__ == "__main__":
    main()
