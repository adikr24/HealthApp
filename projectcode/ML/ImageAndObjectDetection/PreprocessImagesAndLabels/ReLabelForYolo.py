### preprocess the labels to remap to union with current yolo classes ###
### The object classes will stack on top of COCO-80, and some placeholders for hand states ###

# remap_to_union93_safe.py
from pathlib import Path
from collections import Counter
import shutil, time

# ---- EDIT: your split dataset root (has labels/train, labels/val)
ROOT = Path('/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/iterationI/split_dataset')
label_root = ROOT / "labels"
assert label_root.exists(), f"labels/ not found under {ROOT}"

# Try to auto-load the true export order
CANDIDATES = [
    ROOT / 'classes.txt',
    ROOT.parent / 'classes.txt',
    ROOT.parent.parent / 'classes.txt',
]
CURRENT_NAMES = None
for c in CANDIDATES:
    if c.exists():
        CURRENT_NAMES = [ln.strip() for ln in c.read_text().splitlines() if ln.strip()]
        print(f"[INFO] Loaded CURRENT_NAMES from {c} (count={len(CURRENT_NAMES)})")
        break

# Fallback if no classes.txt (your XML order) – OK but riskier if dataset was mixed
if CURRENT_NAMES is None:
    CURRENT_NAMES = [
        "hand_open","hand_closed","hand_pinch",
        "palm","finger",
        "bar","sandwich","blueberry","almond","walnut",
        "plate","bowl","chopping_board","blender","container","sink","knife"
    ]
    print("[WARN] classes.txt not found. Using fallback list from XML; "
          "if your export used a different order, mapping may be wrong.")

# COCO-80 (Ultralytics order)
COCO_80 = [
 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
 'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
 'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
 'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
 'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
 'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
 'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]
COCO_IDX = {n:i for i,n in enumerate(COCO_80)}

# Your true object additions (map to 80..89)
NEW_CLASSES = ["palm","finger","bar","blueberry","almond","walnut","plate","chopping_board","blender","container"]
NEW_IDX = {n: 80 + i for i, n in enumerate(NEW_CLASSES)}

# Placeholders (90..92)
PLACEHOLDERS = ["hand_open","hand_closed","hand_pinch"]
PH_IDX = {n: 90 + i for i, n in enumerate(PLACEHOLDERS)}

# Pseudo/attribute classes to DROP if they appear in CURRENT_NAMES
IGNORED = {
    "on_plate","in_bowl","on_palm","on_counter","in_blender","in_sink",
    "single","multiple",
    "circular","square","designer",
    "small","wide",
    "plastic","glass","metal","paper","unknown",
}

# Build name -> union-id mapping
name_to_union = {}
for name in CURRENT_NAMES:
    if name in COCO_IDX:
        name_to_union[name] = COCO_IDX[name]
    elif name in NEW_IDX:
        name_to_union[name] = NEW_IDX[name]
    elif name in PH_IDX:
        name_to_union[name] = PH_IDX[name]
    elif name in IGNORED:
        name_to_union[name] = None  # drop
    else:
        # Unknown extra class: suggest dropping to keep schema stable
        print(f"[WARN] Unknown class in CURRENT_NAMES: '{name}'. It will be DROPPED (no mapping).")
        name_to_union[name] = None

print("\n=== CURRENT name -> union93 id (None = dropped) ===")
for i, n in enumerate(CURRENT_NAMES):
    print(f"{i:2d} {n:20s} -> {name_to_union[n]}")

# Quick histogram of existing IDs
from collections import defaultdict
id_counts = Counter()
id_files = defaultdict(list)
for p in label_root.rglob('*.txt'):
    for ln in p.read_text().splitlines():
        if not ln.strip(): continue
        parts = ln.split()
        try:
            cid = int(parts[0])
        except:
            continue
        id_counts[cid] += 1
        if len(id_files[cid]) < 3:
            id_files[cid].append(str(p))
print("\n[INFO] Label ID histogram (first 20):", id_counts.most_common(20))

# Backup labels
ts = time.strftime("%Y%m%d-%H%M%S")
backup_dir = ROOT.parent / f"{ROOT.name}_labels_backup_{ts}"
print(f"[INFO] Backing up labels/ to {backup_dir}")
shutil.copytree(label_root, backup_dir)

# Remap in place (skip out-of-range IDs, drop ignored)
updated_files = 0
skipped_out_of_range = 0
dropped_lines = 0

for p in label_root.rglob('*.txt'):
    lines = p.read_text().splitlines()
    out, changed = [], False
    for ln in lines:
        if not ln.strip(): 
            continue
        parts = ln.split()
        try:
            cur_id = int(parts[0])
        except:
            continue

        if cur_id < 0 or cur_id >= len(CURRENT_NAMES):
            skipped_out_of_range += 1
            # keep line unchanged (or comment it out)
            # out.append(ln)   # uncomment if you prefer to keep
            # For correctness, we drop it:
            changed = True
            continue

        cur_name = CURRENT_NAMES[cur_id]
        new_id = name_to_union.get(cur_name, None)

        if new_id is None:
            # drop attributes/unknowns
            dropped_lines += 1
            changed = True
            continue

        if str(new_id) != parts[0]:
            parts[0] = str(new_id)
            changed = True
        out.append(" ".join(parts))

    if changed:
        p.write_text("\n".join(out) + ("\n" if out else ""))
        updated_files += 1

print(f"\n[OK] Remap complete. Updated {updated_files} files.")
print(f"[INFO] Dropped {dropped_lines} attribute/unknown lines; skipped {skipped_out_of_range} out-of-range IDs.")
print("Backup of original labels at:", backup_dir)
print("Next: train with union93.yaml (COCO80 + NEW10 + hand_* 3).")
