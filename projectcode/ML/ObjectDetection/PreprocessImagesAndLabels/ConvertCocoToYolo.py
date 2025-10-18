#!/usr/bin/env python3
# Convert Label Studio COCO → YOLOv5/YOLOv8 (boxes), with polygon fallback and attributes CSV.
# Windows-friendly hardlink/copy; robust path resolution.

import json, shutil, os, errno, csv
from pathlib import Path
from collections import defaultdict

# ========= EDIT THESE =========
ROOT       = Path('/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/IterationII')
COCO_JSON  = ROOT / "result.json"
IMAGES_DIR = ROOT / "images"
OUT_DIR    = ROOT / "yolo_converted"   # will create: images/ + labels/ + classes.txt + attributes.csv

# If True, prefer polygon-derived box when both bbox and segmentation exist
PREFER_POLY_BOX = True
# =================================

# Classes exactly as in your latest Label Studio config
OBJECT_CLASSES = [
    # Hands
    "hand_open", "hand", "hand_fist", "palm", "finger",
    # Kitchen / foods
    "bar", "sandwich", "blueberry", "almond", "walnut",
    "plate", "bowl", "chopping_board", "blender", "container", "sink",
    "knife", "fork", "spoon", "scoop", "cup", "bottle", "lid",
]

# Attribute-like categories (if they accidentally appear as "categories" in the export → ignore as objects)
IGNORE_NAMES = {
    "on_plate", "in_bowl", "on_palm", "on_counter", "in_blender", "in_sink",
    "single", "multiple",
    "circular", "square", "designer",
    "small", "wide",
    "plastic", "glass", "metal", "paper", "unknown",
    "teaspoon", "tablespoon",
}

def find_image(file_name: str, images_dir: Path, root: Path):
    """Normalize backslashes; try direct, relative, and deep search under ROOT."""
    norm = file_name.replace("\\", "/")
    base = Path(norm).name
    p1 = images_dir / base
    if p1.exists():
        return p1
    p2 = images_dir / norm
    if p2.exists():
        return p2
    hits = list(root.rglob(base))
    if hits:
        return hits[0]
    return None

def place_image(src: Path, dst: Path):
    """Create image at dst via hardlink; fallback to copy on cross-device/perm errors."""
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

def poly_to_bbox(segmentation):
    """
    COCO 'segmentation' can be:
      - list of lists of [x1,y1, x2,y2, ...] (polygons)
      - RLE dict (rare from Label Studio; we skip)
    Return (x_min, y_min, w, h) or None if not usable.
    """
    if not segmentation:
        return None
    # polygons case
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
        xs, ys = [], []
        for poly in segmentation:
            # flat list: x1,y1,x2,y2,...
            pts = poly
            if len(pts) >= 6:
                xs.extend(pts[0::2])
                ys.extend(pts[1::2])
        if xs and ys:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            return (float(xmin), float(ymin), float(max(0.0, xmax - xmin)), float(max(0.0, ymax - ymin)))
        return None
    # RLE or other → not handled
    return None

def clamp01(v):  # keep within [0,1]
    return min(max(v, 0.0), 1.0)

def main():
    assert COCO_JSON.exists(), f"Missing {COCO_JSON}"
    assert IMAGES_DIR.exists(), f"Missing {IMAGES_DIR}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)

    coco = json.loads(COCO_JSON.read_text())
    imgs = {im["id"]: im for im in coco.get("images", [])}
    cats = {c["id"]: c for c in coco.get("categories", [])}

    kept_cat_ids, ignored_cat_ids, unknown_cat_ids = set(), set(), set()
    for cid, c in cats.items():
        name = c.get("name", "")
        if name in OBJECT_CLASSES:
            kept_cat_ids.add(cid)
        elif name in IGNORE_NAMES:
            ignored_cat_ids.add(cid)
        else:
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

    present_set = {cats[cid]["name"] for cid in kept_cat_ids}
    present_object_names = [n for n in OBJECT_CLASSES if n in present_set]

    name_to_yolo = {n: i for i, n in enumerate(present_object_names)}
    (OUT_DIR / "classes.txt").write_text("\n".join(present_object_names) + "\n")
    print(f"\n[OK] Wrote classes.txt with {len(present_object_names)} classes → {OUT_DIR/'classes.txt'}")

    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("category_id") in kept_cat_ids:
            anns_by_img[ann["image_id"]].append(ann)

    # Prepare attributes CSV
    attr_csv_path = OUT_DIR / "attributes.csv"
    attr_fieldnames = [
        "image_file", "width", "height",
        "class_name", "class_id",
        "x_center", "y_center", "width_norm", "height_norm",  # YOLO normalized
        "bbox_abs_x", "bbox_abs_y", "bbox_abs_w", "bbox_abs_h",  # absolute px
        # common COCO/LS extras if present
        "area", "iscrowd", "occluded", "truncated",
        # dump-any attributes dict as JSON
        "attributes_json",
        "annotation_id",
    ]
    attr_writer = csv.DictWriter(open(attr_csv_path, "w", newline="", encoding="utf-8"), fieldnames=attr_fieldnames)
    attr_writer.writeheader()

    converted, skipped_missing = 0, 0
    for img_id, im in imgs.items():
        file_name = im.get("file_name", "")
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            continue

        src = find_image(file_name, IMAGES_DIR, ROOT)
        if src is None:
            skipped_missing += 1
            continue

        dst = OUT_DIR / "images" / Path(src).name
        _mode = place_image(src, dst)

        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            name = cats[cid]["name"]
            if name not in name_to_yolo:
                continue
            cls = name_to_yolo[name]

            # Choose bbox source: polygon-derived or provided bbox
            bbox = None
            if PREFER_POLY_BOX:
                bbox = poly_to_bbox(ann.get("segmentation"))
            if bbox is None:
                bbox = ann.get("bbox")
            if not bbox:
                continue

            x, y, bw, bh = map(float, bbox)
            if bw <= 0 or bh <= 0:
                continue

            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h

            cx, cy, nw, nh = map(clamp01, (cx, cy, nw, nh))
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Write attributes row (best-effort; LS adds custom fields under various keys)
            attributes_blob = ann.get("attributes") or ann.get("attrs") or {}
            # capture a few common flags if present
            row = {
                "image_file": dst.name,
                "width": w,
                "height": h,
                "class_name": name,
                "class_id": cls,
                "x_center": f"{cx:.6f}",
                "y_center": f"{cy:.6f}",
                "width_norm": f"{nw:.6f}",
                "height_norm": f"{nh:.6f}",
                "bbox_abs_x": f"{x:.2f}",
                "bbox_abs_y": f"{y:.2f}",
                "bbox_abs_w": f"{bw:.2f}",
                "bbox_abs_h": f"{bh:.2f}",
                "area": ann.get("area", ""),
                "iscrowd": ann.get("iscrowd", 0),
                "occluded": attributes_blob.get("occluded", ""),
                "truncated": attributes_blob.get("truncated", ""),
                "attributes_json": json.dumps(attributes_blob, ensure_ascii=False),
                "annotation_id": ann.get("id", ""),
            }
            attr_writer.writerow(row)

        # Always write a label file (even if empty), YOLO expects it present
        (OUT_DIR / "labels" / (dst.stem + ".txt")).write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
        )
        converted += 1

    print(f"\n[OK] Converted {converted} images → YOLO at: {OUT_DIR}")
    if skipped_missing:
        print(f"[WARN] {skipped_missing} images were referenced in JSON but not found on disk.")

    print("\nArtifacts:")
    print(f"  - {OUT_DIR}/images/")
    print(f"  - {OUT_DIR}/labels/")
    print(f"  - {OUT_DIR}/classes.txt")
    print(f"  - {attr_csv_path}")
    print("\nTrain tip:")
    print("  Point your data.yaml to these images/labels. Use attributes.csv for ROI state classifiers or QA checks.")

if __name__ == "__main__":
    main()
