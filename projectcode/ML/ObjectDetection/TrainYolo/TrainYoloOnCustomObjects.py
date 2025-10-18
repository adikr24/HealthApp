#!/usr/bin/env python3
# TrainYoloOnCustomObjects.py
# Python 3.9+; Ultralytics YOLOv8
# Purpose: fail-fast env checks, validate data.yaml, then train @ 640 on YOLOv8n

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# ---- imports at the top (fail fast if missing) ----
import torch
from ultralytics import YOLO
import yaml  # pip install pyyaml



# ---------- config defaults (edit to your paths) ----------
DATA_YAML   = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloIterations/IterationII/config/config.yaml"
WEIGHTS_DIR = "/app/mediaFiles/yolo_settings/YoloWeights/"
DEFAULT_MODEL = "/app/mediaFiles/yolo_settings/YoloWeights/yolov8n.pt"  # force 8n by default
PROJECT_DIR = "/app/mediaFiles/images/YoloTrainingImages/yolotrainingReadyImages/YoloResultIterations/IterationIIResults/"
RUN_NAME    = "union93_yolov8n_640"

IMG_SZ   = 640      # fixed as requested
EPOCHS   = 70
FREEZE   = 10
LR0      = 0.003
WORKERS  = 0        # auto-downgraded to 0 on Windows
SEED     = 42

try:
    # avoid POSIX shared memory so small /dev/shm doesn't crash workers
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    os.environ["PYTORCH_STORAGE_SHARING_STRATEGY"] = "file_system"

# ---------- helpers ----------
def _p(x: str) -> str:
    return os.path.normpath(os.path.expanduser(os.path.expandvars(x)))



def gpu_autosize():
    """
    Return (device_str, total_vram_mb, suggested_batch:int) for imgsz=640.
    Choose a safe numeric batch so newer Ultralytics versions accept it.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory // (1024 * 1024)
        # conservative numeric guesses
        if vram_mb >= 16384:      # >=16 GB
            batch = 24
        elif vram_mb >= 12288:    # 12–15 GB
            batch = 16
        elif vram_mb >= 8192:     # 8–11 GB
            batch = 12
        elif vram_mb >= 6144:     # 6–7 GB
            batch = 8
        else:
            batch = 4
        return "0", vram_mb, batch
    return "cpu", 0, 4

def check_data_yaml(data_yaml: Path):
    if not data_yaml.exists():
        raise FileNotFoundError(f"[data.yaml] not found: {data_yaml}")

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # Required keys
    for key in ("train", "val", "names"):
        if key not in cfg:
            raise ValueError(f"[data.yaml] missing required key: '{key}'")

    # train/val can be list(s) of dirs/files or single string; normalize to lists and existence check
    for split in ("train", "val"):
        items = cfg[split]
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list) or not items:
            raise ValueError(f"[data.yaml] '{split}' must be a non-empty list or string: {data_yaml}")

        for p in items:
            pp = Path(_p(p))
            if not pp.exists():
                raise FileNotFoundError(f"[data.yaml] {split} path does not exist: {pp}")

    # names / nc sanity
    names = cfg["names"]
    if isinstance(names, dict):
        # convert to list by key order if dict-like
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
    if not isinstance(names, list) or not all(isinstance(n, (str, int)) for n in names):
        raise ValueError(f"[data.yaml] 'names' must be a list (or index->name dict). Got: {type(names)}")

    nc = cfg.get("nc", len(names))
    if nc != len(names):
        raise ValueError(f"[data.yaml] nc ({nc}) != len(names) ({len(names)}). They must match.")

    return cfg

def pick_model_path(arg_model: str, weights_dir: Path) -> str:
    """
    Choose the correct YOLO model checkpoint:
      1. If user passes a path (--model), use that.
      2. Else prefer local yolov8n.pt in weights_dir (legacy default).
      3. Else fall back to yolov11n.pt if present.
      4. If neither found, return 'yolov8n.pt' (Ultralytics will download).
    """
    # Normalize and expand all paths
    weights_dir = Path(_p(str(weights_dir)))
    cand_model = Path(_p(arg_model)) if arg_model else None

    # 1️⃣ Explicit model argument
    if cand_model and cand_model.exists():
        print(f"[info] Using user-provided model: {cand_model}")
        return str(cand_model)

    # 2️⃣ Prefer yolov8n.pt
    y8_path = weights_dir / "yolov8n.pt"
    if y8_path.exists():
        print(f"[info] Found local yolov8n.pt → {y8_path}")
        return str(y8_path)

    # 3️⃣ Fall back to yolov11n.pt if present
    y11_path = weights_dir / "yolov11n.pt"
    if y11_path.exists():
        print(f"[info] Found local yolov11n.pt → {y11_path}")
        return str(y11_path)

    # 4️⃣ Last resort: let Ultralytics fetch if missing
    print("[warn] Neither yolov8n.pt nor yolov11n.pt found locally; Ultralytics will download.")
    return "yolov8n.pt"


def is_windows():
    return sys.platform.startswith("win")

def save_run_summary(project: Path, name: str, summary: dict):
    out_dir = project / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] Wrote run summary -> {summary_path}")

# ---------- main ----------
def main(argv=None):
    parser = argparse.ArgumentParser("Train YOLOv8n at 640 on your custom set (transfer from COCO)")
    parser.add_argument("--data", type=str, default=DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--weights_dir", type=str, default=WEIGHTS_DIR, help="Folder with weights")
    parser.add_argument("--model", type=str, default="", help="Model path or name (default yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", default="", help="Batch size (int or 'auto'); default uses GPU heuristic")
    parser.add_argument("--project", type=str, default=PROJECT_DIR)
    parser.add_argument("--name", type=str, default=RUN_NAME)
    parser.add_argument("--device", type=str, default="", help="'0', 'cpu', etc.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze", type=int, default=FREEZE, help="Number of backbone layers to freeze")
    parser.add_argument("--lr0", type=float, default=LR0, help="Initial learning rate")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for speed (needs memory)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--export_onnx", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    # deterministic run
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_yaml = Path(_p(args.data))
    weights_dir = Path(_p(args.weights_dir))
    project = Path(_p(args.project))

    # Device & batch (imgsz fixed = 640)
    device_auto, vram_mb, batch_suggest = gpu_autosize()
    device = args.device or device_auto
    batch = args.batch or batch_suggest
    try:
        batch = int(batch) if str(batch).isdigit() else batch
    except Exception:
        pass

    # workers: Windows tends to hang with >0 sometimes
    workers = 0 if is_windows() else WORKERS

    # Env print
    print("=========== ENV CHECK ===========")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (MB):", vram_mb)
    print("Using device:", device, "| imgsz:", IMG_SZ, "| batch:", batch, "| workers:", workers)
    print("=================================")

    # Data sanity
    cfg = check_data_yaml(data_yaml)

    # Model path (force v8n by default)
    model_path = pick_model_path(args.model, weights_dir)
    print("Model to load:", model_path)

    # Load model
    model = YOLO(model_path)

    # Optional: disable W&B
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=IMG_SZ,
        batch=batch,
        device=device,
        freeze=args.freeze,
        lr0=args.lr0,
        cos_lr=True,
        close_mosaic=10,
        project=str(project),
        name=args.name,
        workers=workers,
        cache="ram" if args.cache else False,
        seed=args.seed,
        deterministic=True,
    )
    if args.resume:
        train_kwargs["resume"] = True

    # Save a small run summary up-front
    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(data_yaml),
        "names": cfg.get("names"),
        "nc": cfg.get("nc", len(cfg.get("names", []))),
        "model_init": model_path,
        "imgsz": IMG_SZ,
        "epochs": args.epochs,
        "batch": batch,
        "device": device,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "workers": workers,
        "cache": bool(args.cache),
        "seed": args.seed,
        "wandb_disabled": bool(args.no_wandb),
    }
    save_run_summary(project, args.name, run_summary)

    print("\n===== START TRAIN =====")
    results = model.train(**train_kwargs)
    print("Train done.")

    # Validate best weights
    print("\n===== START VAL =====")
    try:
        # In recent Ultralytics, best path is usually at runs/..../weights/best.pt
        best = getattr(model, "ckpt_path", None)
        if not best:
            best = project / args.name / "weights" / "best.pt"
        best = Path(str(best))
        if not best.exists():
            # fallback to last
            alt = project / args.name / "weights" / "last.pt"
            if alt.exists():
                best = alt
        print("Validating model:", best)
        model = YOLO(str(best))
        model.val(data=str(data_yaml), imgsz=IMG_SZ, device=device, workers=workers)
        print("Val done.")

        if args.export_onnx:
            print("\n===== EXPORT (ONNX) =====")
            model.export(format="onnx", opset=12)  # creates best.onnx next to weights
            print("Exported ONNX.")
    except Exception as e:
        print("[WARN] Validation/export skipped/failed:", e)

    print("\nAll done. See:", project / args.name)

if __name__ == "__main__":
    sys.exit(main())
