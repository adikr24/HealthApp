#!/usr/bin/env python3
"""
*** This code helps download all objects for Yolo to not let it do catastrophic forgetting ***


Prepare a YOLO-format rehearsal set from COCO *val2017* only.

- Downloads ONLY val2017 (+ annotations) into:
    /app/mediaFiles/images/YoloTrainingImages/ImagesYoloTrainedOnAlready/_coco
- Converts to YOLO .txt labels and copies images into:
    /app/mediaFiles/images/YoloTrainingImages/ImagesYoloTrainedOnAlready/{images,labels}/{val}
- Writes a dataset YAML at:
    /app/mediaFiles/images/YoloTrainingImages/ImagesYoloTrainedOnAlready/coco-val-yolo.yaml

SAFE: Does NOT delete the underlying folder. It only clears/rebuilds
the generated 'images/', 'labels/' subfolders and the YAML.

If you want to shrink further, set VAL_KEEP to a smaller number
(e.g., 2000) to randomly sample that many images from val2017 (≈5k total).
"""
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ================== CONFIG (edit here if needed) ==================
ROOT = Path("/app/mediaFiles/images/YoloTrainingImages/ImagesYoloTrainedOnAlready").resolve()
COCO_ROOT = ROOT / "_coco"                 # where zips + raw COCO live
DATA_YAML = ROOT / "coco-val-yolo.yaml"    # output dataset yaml
VAL_KEEP = 5000                            # how many val images to keep (<= 5000). Use 5000 for full val.
SEED = 42                                  # random seed for sampling
# =================================================================

COCO_URLS = {
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "ann":        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

YOLO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# COCO category_id → YOLO name (COCO has 91 IDs; only these 80 are used)
COCO_ID_TO_NAME = {
    1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
    11:'fire hydrant',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',21:'cow',
    22:'elephant',23:'bear',24:'zebra',25:'giraffe',27:'backpack',28:'umbrella',31:'handbag',32:'tie',33:'suitcase',34:'frisbee',
    35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',41:'skateboard',42:'surfboard',43:'tennis racket',
    44:'bottle',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',
    56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',
    67:'dining table',70:'toilet',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',
    80:'toaster',81:'sink',82:'refrigerator',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush'
}
NAME_TO_YOLO_IDX = {n: i for i, n in enumerate(YOLO_NAMES)}
COCO_ID_TO_YOLO_IDX = {cid: NAME_TO_YOLO_IDX[name] for cid, name in COCO_ID_TO_NAME.items()}

def download_with_progress(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit='B', unit_scale=True, desc=out_path.name) as pbar:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def unzip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)

def ensure_coco_val_present(coco_root: Path):
    """Ensure ONLY val2017 images and annotations are present locally."""
    zips_dir = coco_root / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    vzip = zips_dir / "val2017.zip"
    azip = zips_dir / "annotations_trainval2017.zip"

    # Download only val + annotations
    download_with_progress(COCO_URLS["val_images"], vzip)
    download_with_progress(COCO_URLS["ann"], azip)

    # Unzip (idempotent)
    if not (coco_root / "val2017").exists():
        unzip(vzip, coco_root)
    if not (coco_root / "annotations").exists():
        unzip(azip, coco_root)

    # Sanity check
    required = [
        coco_root / "val2017",
        coco_root / "annotations" / "instances_val2017.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"COCO val not found or incomplete. Missing: {missing}")
    return coco_root

def pick_random_val_ids(val_json: Path, keep_n: int, seed: int):
    with open(val_json, "r") as f:
        coco = json.load(f)
    image_ids = [img["id"] for img in coco["images"]]
    if keep_n >= len(image_ids):
        return set(image_ids)
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    return set(image_ids[:keep_n])

def coco_to_yolo_txts(coco_json_path: Path, images_dir: Path, out_labels_dir: Path, out_images_dir: Path, keep_image_ids=None):
    """
    Convert a COCO instances JSON to YOLO-format .txt files and copy images.
    If keep_image_ids is provided, only those images are processed.
    """
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    imgs_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        if keep_image_ids is not None and img_id not in keep_image_ids:
            continue
        anns_by_img.setdefault(img_id, []).append(ann)

    image_ids = list(imgs_by_id.keys())
    if keep_image_ids is not None:
        image_ids = [i for i in image_ids if i in keep_image_ids]

    created = []
    for img_id in tqdm(image_ids, desc=f"Converting {coco_json_path.stem}"):
        img_info = imgs_by_id[img_id]
        file_name = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        src_img = images_dir / file_name
        if not src_img.exists():
            continue

        dst_img = out_images_dir / file_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        label_path = out_labels_dir / (Path(file_name).stem + ".txt")
        anns = anns_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            cid = ann["category_id"]
            if cid not in COCO_ID_TO_YOLO_IDX:
                continue
            cls = COCO_ID_TO_YOLO_IDX[cid]
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            if nw <= 0 or nh <= 0:
                continue
            # clamp (paranoia)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as lf:
            lf.write("\n".join(lines))
        created.append(file_name)

    return created

def write_yaml(out_yaml: Path, root_dir: Path):
    text = f"""# COCO val2017 rehearsal set (YOLO format)
# Generated for continual learning (prevent forgetting) with minimal download.
path: {root_dir.as_posix()}
train:
  - images/val   # we use val as both train/val for rehearsal; adjust if you prefer
val:
  - images/val
names:
{os.linesep.join([f"  - {n}" for n in YOLO_NAMES])}
"""
    out_yaml.write_text(text)

def safe_rebuild_generated_dirs(root: Path):
    """
    Clears only the generated subfolders (images/, labels/) and YAML,
    leaving _coco/ untouched.
    """
    targets = [root / "images", root / "labels", DATA_YAML]
    for t in targets:
        if t.exists():
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()

def main():
    # 1) Ensure only val is present (download if needed)
    ensure_coco_val_present(COCO_ROOT)

    val_json = COCO_ROOT / "annotations" / "instances_val2017.json"
    val_dir  = COCO_ROOT / "val2017"

    # 2) Sample VAL_KEEP images (or keep all if VAL_KEEP >= 5000)
    keep_ids = pick_random_val_ids(val_json, VAL_KEEP, SEED)

    # 3) SAFE cleanup of generated outputs (keep _coco)
    safe_rebuild_generated_dirs(ROOT)

    # 4) Convert/copy to YOLO format
    out_imgs_val = ROOT / "images" / "val"
    out_lbls_val = ROOT / "labels" / "val"
    created_val = coco_to_yolo_txts(
        coco_json_path=val_json,
        images_dir=val_dir,
        out_labels_dir=out_lbls_val,
        out_images_dir=out_imgs_val,
        keep_image_ids=keep_ids
    )

    # 5) Write dataset YAML
    write_yaml(DATA_YAML, ROOT)

    print("\nDone.")
    print(f"Val images created: {len(created_val)}  → {out_imgs_val}")
    print(f"YAML written:       {DATA_YAML}")
    print("\nTraining tip:")
    print("  yolo detect train model=yolov8l.pt data="
          f"{DATA_YAML} epochs=1 imgsz=640")
    print("For your combined training, list this dataset under `train:` along with your 3-class set.")

if __name__ == "__main__":
    main()
