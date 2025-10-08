## convert the COCO annotations to YOLOv5/YOLOv8 format (robust path handling)
## If you download file from label studio and if it's in COCO format, you can use this script to convert it to YOLO format. ##
import json, shutil
from pathlib import Path
from collections import defaultdict

# ========= EDIT THESE =========
ROOT = Path('/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/iterationII')
COCO_JSON = ROOT / "result.json"
IMAGES_DIR = ROOT / "images"
OUT_DIR = ROOT / "yolo_converted"          # will create: images/ (symlinks or copies) + labels/
# =================================

# Object classes you actually want boxes for (case-sensitive, match your Label Studio names)
OBJECT_CLASSES = {
    "palm", "finger",
    "bar", "sandwich", "blueberry", "almond", "walnut",
    "plate", "bowl", "chopping_board", "blender", "container", "sink", "knife",
    # add any other true objects you boxed:
    # "water_filter", ...
}

# Attribute / context names to IGNORE if they appear as categories
IGNORE_NAMES = {
    "on_plate", "in_bowl", "on_palm", "on_counter", "in_blender", "in_sink",
    "single", "multiple",
    "circular", "square", "designer",
    "small", "wide",
    "plastic", "glass", "metal", "paper", "unknown",
}

def find_image(file_name: str, images_dir: Path, root: Path) :
    """
    Make Label Studio / COCO file_name robust:
    - Normalize backslashes -> '/'
    - Use basename to match files exported elsewhere
    - Try images_dir/<basename>, then images_dir/<normalized>, then search under ROOT
    """
    norm = file_name.replace("\\", "/")
    base = Path(norm).name

    p1 = images_dir / base
    if p1.exists():
        return p1

    p2 = images_dir / norm
    if p2.exists():
        return p2

    # last resort: search anywhere under ROOT (slower but reliable)
    hits = list(root.rglob(base))
    if hits:
        return hits[0]

    return None

def main():
    assert COCO_JSON.exists(), f"Missing {COCO_JSON}"
    assert IMAGES_DIR.exists(), f"Missing {IMAGES_DIR}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    coco = json.loads(COCO_JSON.read_text())

    # COCO dicts
    imgs = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c for c in coco["categories"]}

    # Split categories into KEPT vs IGNORED
    kept_cat_ids, ignored_cat_ids, unknown_cat_ids = set(), set(), set()
    for cid, c in cats.items():
        name = c["name"]
        if name in OBJECT_CLASSES:
            kept_cat_ids.add(cid)
        elif name in IGNORE_NAMES:
            ignored_cat_ids.add(cid)
        else:
            # treat anything else as "unknown attribute" (ignore by default)
            unknown_cat_ids.add(cid)

    print("=== Category summary ===")
    print("Kept object classes:")
    for cid in sorted(kept_cat_ids):
        print(f"  - {cats[cid]['name']} (id {cid})")
    print("\nIgnored attribute-like classes:")
    for cid in sorted(ignored_cat_ids):
        print(f"  - {cats[cid]['name']} (id {cid})")
    if unknown_cat_ids:
        print("\nOther categories (ignored by default):")
        for cid in sorted(unknown_cat_ids):
            print(f"  - {cats[cid]['name']} (id {cid})")

    # Build YOLO class list (order = OBJECT_CLASSES that actually appear)
    present_set = {cats[cid]["name"] for cid in kept_cat_ids}
    present_object_names = [n for n in OBJECT_CLASSES if n in present_set]

    name_to_yolo = {n: i for i, n in enumerate(present_object_names)}
    (OUT_DIR / "classes.txt").write_text("\n".join(present_object_names) + "\n")
    print(f"\n[OK] Wrote classes.txt with {len(present_object_names)} classes at {OUT_DIR/'classes.txt'}")

    # Group annotations by image (kept objects only)
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] in kept_cat_ids:
            anns_by_img[ann["image_id"]].append(ann)

    # Convert to YOLO
    converted, skipped_missing = 0, 0
    for img_id, im in imgs.items():
        file_name = im["file_name"]
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            # (rare) if width/height missing, skip safely
            continue

        src = find_image(file_name, IMAGES_DIR, ROOT)
        if src is None:
            skipped_missing += 1
            continue

        # Put the found image into OUT_DIR/images using its basename
        dst = OUT_DIR / "images" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)  # fast & storage-light
        except Exception:
            shutil.copy2(src, dst)  # Windows/WSL fallback

        # Write YOLO labels
        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            name = cats[cid]["name"]
            if name not in name_to_yolo:
                continue
            cls = name_to_yolo[name]
            x, y, bw, bh = ann["bbox"]  # COCO absolute xywh
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            # clip & validate
            if nw <= 0 or nh <= 0:
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (OUT_DIR / "labels" / (dst.stem + ".txt")).write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
        )
        converted += 1

    print(f"\n[OK] Converted {converted} images to YOLO under {OUT_DIR}")
    if skipped_missing:
        print(f"[WARN] {skipped_missing} images were referenced in JSON but not found on disk after normalization/search.")
    print("\nNext:")
    print(f"  - Train directly with data pointing to {OUT_DIR}/images & {OUT_DIR}/labels")
    print("  - Or run the union mapping later to align with COCO80+custom in one model.")

if __name__ == "__main__":
    main()
