#!/usr/bin/env python3
# build_coco_kitchen.py
from pathlib import Path
from typing import List, Dict, Set
import json, zipfile, random, shutil
import requests
from tqdm import tqdm
from collections import defaultdict

# ---------- CONFIG ----------
ROOT = Path("/app/mediaFiles/datasets/COCO_Kitchen").resolve()
DOWNLOADS = ROOT / "_raw"
OUT = ROOT / "yolo"
INCLUDE_VAL = True                 # also include val2017 as replay
MAX_IMAGES_PER_CLASS = 1500        # None for all; or small balanced replay (e.g., 500–2000)
SEED = 42

# Choose only the COCO classes you care about (subset of COCO80)
KITCHEN_CLASS_NAMES = [
    "bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot",
    "hot dog","pizza","donut","cake",
    "microwave","oven","toaster","refrigerator","sink","dining table"
]

# Full COCO80 names in YOLO order (Ultralytics)
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
NAME_TO_YOLO_IDX = {n:i for i,n in enumerate(YOLO_NAMES)}

# COCO category_id → YOLO name (only 80 used)
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
# ---------------------------

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images":   "http://images.cocodataset.org/zips/val2017.zip",
    "ann":          "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def dl(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm(total=total, unit='B', unit_scale=True, desc=out.name) as p:
            with open(out, "wb") as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk:
                        f.write(chunk); p.update(len(chunk))

def unzip(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)

def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def build_subset(split: str, images_dir: Path, ann_json: Dict, keep_names: Set[str],
                 out_img_dir: Path, out_lbl_dir: Path,
                 max_per_class: int | None, rng: random.Random):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    cats = {c["id"]: c["name"] for c in ann_json["categories"]}
    keep_cat_ids = {cid for cid, nm in cats.items() if nm in keep_names}
    # image_id -> list of kept anns
    anns_by_img = defaultdict(list)
    for a in ann_json["annotations"]:
        if a["category_id"] in keep_cat_ids and a.get("iscrowd",0)==0 and a["bbox"][2]>1 and a["bbox"][3]>1:
            anns_by_img[a["image_id"]].append(a)

    # per-class buckets (image ids)
    per_class_imgs = defaultdict(set)
    for img in ann_json["images"]:
        kept = anns_by_img.get(img["id"], [])
        cls_set = {COCO_ID_TO_NAME[a["category_id"]] for a in kept}
        for cname in cls_set:
            if cname in keep_names:
                per_class_imgs[cname].add(img["id"])

    # choose subset if max_per_class is set
    selected_imgs: Set[int] = set()
    if max_per_class:
        for cname, ids in per_class_imgs.items():
            ids_list = list(ids)
            rng.shuffle(ids_list)
            selected_imgs.update(ids_list[:max_per_class])
    else:
        for ids in per_class_imgs.values():
            selected_imgs.update(ids)

    # Build quick image dict
    img_by_id = {i["id"]: i for i in ann_json["images"]}

    # Export
    kept_count = 0
    for img_id in selected_imgs:
        img = img_by_id[img_id]
        src = images_dir / img["file_name"]
        if not src.exists():
            continue
        dst = out_img_dir / img["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

        # YOLO label file
        lbl = out_lbl_dir / (Path(img["file_name"]).stem + ".txt")
        w, h = img["width"], img["height"]
        lines = []
        for a in anns_by_img.get(img_id, []):
            cname = COCO_ID_TO_NAME[a["category_id"]]
            if cname not in keep_names: 
                continue
            # YOLO class id: reindex to a compact kitchen-only range
            yolo_local_id = sorted(list(keep_names)).index(cname)
            # bbox: [x,y,w,h] -> YOLO [cx,cy,w,h] normalized
            x, y, bw, bh = a["bbox"]
            cx, cy = x + bw/2, y + bh/2
            lines.append(f"{yolo_local_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        if lines:
            with open(lbl, "w") as f:
                f.write("\n".join(lines))
            kept_count += 1
    print(f"[{split}] wrote {kept_count} labeled images to {out_img_dir}")

def write_yaml(yaml_path: Path, train_dir: Path, val_dir: Path, class_names: List[str]):
    yaml = [
        f"path: {yaml_path.parent}",
        f"train: {train_dir.relative_to(yaml_path.parent)}",
        f"val: {val_dir.relative_to(yaml_path.parent)}",
        "names:"
    ]
    for i, n in enumerate(class_names):
        yaml.append(f"  {i}: {n}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml))
    print(f"Wrote {yaml_path}")

def main():
    rng = random.Random(SEED)
    # downloads
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    dl(COCO_URLS["train_images"], DOWNLOADS/"train2017.zip")
    if INCLUDE_VAL:
        dl(COCO_URLS["val_images"], DOWNLOADS/"val2017.zip")
    dl(COCO_URLS["ann"], DOWNLOADS/"annotations_trainval2017.zip")

    # unzip
    unzip(DOWNLOADS/"train2017.zip", DOWNLOADS)
    if INCLUDE_VAL:
        unzip(DOWNLOADS/"val2017.zip", DOWNLOADS)
    unzip(DOWNLOADS/"annotations_trainval2017.zip", DOWNLOADS)

    # paths
    train_imgs = DOWNLOADS/"train2017"
    val_imgs   = DOWNLOADS/"val2017"
    ann_train  = json.load(open(DOWNLOADS/"annotations"/"instances_train2017.json"))
    ann_val    = json.load(open(DOWNLOADS/"annotations"/"instances_val2017.json"))

    # out dirs
    train_out_img = OUT/"images"/"train"
    val_out_img   = OUT/"images"/"val"
    train_out_lbl = OUT/"labels"/"train"
    val_out_lbl   = OUT/"labels"/"val"

    keep_set = set(KITCHEN_CLASS_NAMES)

    build_subset("train", train_imgs, ann_train, keep_set, train_out_img, train_out_lbl,
                 MAX_IMAGES_PER_CLASS, rng)
    if INCLUDE_VAL:
        build_subset("val", val_imgs, ann_val, keep_set, val_out_img, val_out_lbl,
                     MAX_IMAGES_PER_CLASS, rng)
    else:
        # if not using val, create a tiny val from train
        val_out_img.mkdir(parents=True, exist_ok=True)
        val_out_lbl.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: names are reindexed to the *kitchen-only* compact range
    names_compact = sorted(list(keep_set))
    write_yaml(OUT/"coco_kitchen.yaml", OUT/"images/train", OUT/"images/val", names_compact)

if __name__ == "__main__":
    main()
