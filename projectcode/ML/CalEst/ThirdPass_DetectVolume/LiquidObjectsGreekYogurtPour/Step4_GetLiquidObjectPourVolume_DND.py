#!/usr/bin/env python3
# Frame-by-frame white (yogurt) differencing on band crops
# - Loads band_crops from greek_yogurt OUT_DIR
# - Per frame: luminance -> white mask, save blue→red heatmap of luminance
# - Per pair: XOR change of white mask, save blue→red change heatmap
# - CSV with white counts and deltas

from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------- Paths (adjust if needed) ----------
OUT_DIR         = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/greek_yogurt/")
BAND_CROPS_DIR  = OUT_DIR / "band_crops"     # expects files like crop_00494.png
HEAT_DIR        = OUT_DIR / "gray_heat"      # per-frame luminance heatmaps
DIFF_DIR        = OUT_DIR / "diff_heat"      # per-pair change heatmaps
CSV_OUT         = OUT_DIR / "band_diff_summary.csv"

# ---------- Knobs ----------
USE_CLAHE   = True
GAMMA       = 0.8
WHITE_THR   = 230   # luminance threshold for "thick white" yogurt pixels
OUTPUT_SCALE = 4.0

# ---------- Helpers ----------
def idx_from_fn(fn: str) -> int:
    m = re.search(r"(\d+)", Path(fn).stem)
    return int(m.group(1)) if m else -1

def imread_any(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None and path.exists() and path.stat().st_size > 0:
        try:
            with Image.open(str(path)) as im:
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img

def gamma_LUT(gamma=0.8):
    x = np.arange(256, dtype=np.float32)/255.0
    y = np.power(x, gamma)
    return (np.clip(y*255,0,255)).astype(np.uint8)

def build_blue2red_LUT():
    lut = np.zeros((256,1,3), dtype=np.uint8)
    for i in range(256):
        t = i/255.0
        b = int(round(255*(1.0 - t)))
        r = int(round(255*t))
        lut[i,0,:] = (b,0,r)
    return lut

_GAMMA_LUT   = gamma_LUT(GAMMA)
_CLAHE       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None
_BLUE2RED_LUT = build_blue2red_LUT()

def to_gray_lum(bgr: np.ndarray) -> np.ndarray:
    """Lab-L channel with optional CLAHE + gamma for robust luminance."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    if _CLAHE is not None:
        L = _CLAHE.apply(L)
    return cv2.LUT(L, _GAMMA_LUT)

def colorize_heat(gray_0to255: np.ndarray) -> np.ndarray:
    g = np.clip(gray_0to255, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), _BLUE2RED_LUT)

def white_mask_from_gray(gray: np.ndarray, thr: int) -> np.ndarray:
    return (gray >= thr).astype(np.uint8) * 255  # {0,255}

def save_scaled(img: np.ndarray, path: Path, scale: float = OUTPUT_SCALE):
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)

# ---------- Main ----------
def main():
    HEAT_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Collect band crops (png/jpg) sorted by index
    crops = list(BAND_CROPS_DIR.glob("crop_*.png")) + list(BAND_CROPS_DIR.glob("crop_*.jpg"))
    crops.sort(key=lambda p: idx_from_fn(p.name))
    if not crops:
        print(json.dumps({"error":"no band crops found", "dir": str(BAND_CROPS_DIR)}, indent=2))
        return

    rows = []
    prev_idx, prev_mask = None, None
    prev_name = None

    for cp in crops:
        idx = idx_from_fn(cp.name)
        img = imread_any(cp)
        if img is None:
            rows.append({"frame": cp.name, "frame_idx": idx, "error": "failed to read crop"})
            continue

        # 2) Luminance + per-frame heatmap (for visual sanity)
        L = to_gray_lum(img)
        heat = colorize_heat(L)
        heat_path = HEAT_DIR / f"gray_heat_{idx:05d}.png"
        save_scaled(heat, heat_path)

        # 3) White mask (thick yogurt)
        wmask = white_mask_from_gray(L, WHITE_THR)
        white_count = int((wmask > 0).sum())
        H, W = wmask.shape[:2]
        white_ratio = white_count / float(H*W) if H*W > 0 else 0.0

        # 4) Frame-to-frame change (XOR of white masks)
        delta_white = None
        change_count = None
        diff_path = ""

        if prev_mask is not None:
            # If dims change, resize current mask to prev dims for fair XOR
            cur_mask = wmask
            if cur_mask.shape != prev_mask.shape:
                cur_mask = cv2.resize(cur_mask, (prev_mask.shape[1], prev_mask.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            xor = cv2.absdiff(cur_mask, prev_mask)           # 0 or 255 where different
            change_count = int((xor > 0).sum())
            delta_white = white_count - int((prev_mask > 0).sum())

            # Colorize XOR for readability
            xor_big = cv2.resize(xor, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
            xor_heat = colorize_heat(xor_big)
            diff_path = str(DIFF_DIR / f"white_change_{prev_idx:05d}_{idx:05d}.png")
            cv2.imwrite(diff_path, xor_heat)

        rows.append({
            "frame": cp.name,
            "frame_idx": idx,
            "gray_heat": str(heat_path),
            "white_thr": WHITE_THR,
            "white_count": white_count,
            "white_ratio": round(white_ratio, 6),
            "prev_frame": prev_name or "",
            "delta_white": int(delta_white) if delta_white is not None else "",
            "change_pixels": int(change_count) if change_count is not None else "",
            "white_change_heatmap": diff_path,
        })

        prev_mask = wmask
        prev_idx = idx
        prev_name = cp.name

    # 5) Save CSV
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    print(json.dumps({
        "done": True,
        "band_crops_dir": str(BAND_CROPS_DIR),
        "gray_heat_dir": str(HEAT_DIR),
        "diff_heat_dir": str(DIFF_DIR),
        "csv": str(CSV_OUT),
        "frames": len(rows),
        "white_thr": WHITE_THR
    }, indent=2))

if __name__ == "__main__":
    main()
