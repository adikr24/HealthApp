#!/usr/bin/env python3
# Protein-powder pixel differencing on existing band_crops
# Reads band_crops under .../volume_bands/<ooi>/band_crops
# Writes gray_heat/, diff_heat/ and a CSV next to that <ooi> folder.

from __future__ import annotations
import re, json
from pathlib import Path
from typing import Optional
import cv2, numpy as np, pandas as pd
from PIL import Image

# ---------- Point this to your band_crops dir ----------
BAND_CROPS_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/protein_powder/band_crops")

# Derive OUT_DIR and outputs from band_crops location
OUT_DIR  = BAND_CROPS_DIR.parent
HEAT_DIR = OUT_DIR / "gray_heat"
DIFF_DIR = OUT_DIR / "diff_heat"
CSV_OUT  = OUT_DIR / "band_diff_summary_proteinpowder.csv"

# ---------- Knobs ----------
USE_CLAHE    = True
GAMMA        = 0.9
OUTPUT_SCALE = 4.0

# Powder color gating
L_MIN       = 170
A_DEV       = 18
B_DEV_POS   = 32
B_DEV_NEG   = 20
S_MAX       = 60

# Morphology / noise control
DO_MORPH        = True
MORPH_OPEN_K    = 3
MORPH_CLOSE_K   = 5
MIN_BLOB_AREA   = 12

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

def gamma_LUT(gamma=0.9):
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

_GAMMA_LUT    = gamma_LUT(GAMMA)
_CLAHE        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None
_BLUE2RED_LUT = build_blue2red_LUT()

def to_gray_lum(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    if _CLAHE is not None:
        L = _CLAHE.apply(L)
    return cv2.LUT(L, _GAMMA_LUT)

def colorize_heat(gray_0to255: np.ndarray) -> np.ndarray:
    g = np.clip(gray_0to255, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), _BLUE2RED_LUT)

def powder_mask_from_bgr(bgr: np.ndarray) -> np.ndarray:
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L    = lab[...,0].astype(np.int16)
    a    = lab[...,1].astype(np.int16)
    b    = lab[...,2].astype(np.int16)
    a_c  = np.abs(a - 128) <= A_DEV
    b_c  = (b - 128 >= -B_DEV_NEG) & (b - 128 <= B_DEV_POS)
    L_c  = L >= L_MIN

    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S    = hsv[...,1].astype(np.int16)
    S_c  = S <= S_MAX

    mask = (L_c & a_c & b_c & S_c).astype(np.uint8) * 255

    if DO_MORPH:
        if MORPH_OPEN_K > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if MORPH_CLOSE_K > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        num, labels, stats, _ = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), 8)
        if num > 1:
            keep = np.zeros_like(mask, dtype=np.uint8)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_AREA:
                    keep[labels == i] = 255
            mask = keep
    return mask

def save_scaled(img: np.ndarray, path: Path, scale: float = OUTPUT_SCALE):
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)

# ---------- Main ----------
def main():
    HEAT_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR.mkdir(parents=True, exist_ok=True)

    crops = sorted(
        list(BAND_CROPS_DIR.glob("crop_*.png")) + list(BAND_CROPS_DIR.glob("crop_*.jpg")),
        key=lambda p: idx_from_fn(p.name)
    )
    if not crops:
        print(json.dumps({"error":"no band crops found", "dir": str(BAND_CROPS_DIR)}, indent=2))
        return

    rows = []
    prev_idx = None
    prev_mask = None
    prev_name = None

    for cp in crops:
        idx = idx_from_fn(cp.name)
        img = imread_any(cp)
        if img is None:
            rows.append({"frame": cp.name, "frame_idx": idx, "error": "failed to read crop"})
            continue

        L = to_gray_lum(img)
        heat = colorize_heat(L)
        heat_path = HEAT_DIR / f"gray_heat_{idx:05d}.png"
        save_scaled(heat, heat_path)

        pmask = powder_mask_from_bgr(img)
        powder_count = int((pmask > 0).sum())
        H, W = pmask.shape[:2]
        powder_ratio = powder_count / float(H*W) if H*W else 0.0

        delta_powder = ""
        change_pixels = ""
        diff_path = ""

        if prev_mask is not None:
            cur_mask = pmask if pmask.shape == prev_mask.shape else cv2.resize(
                pmask, (prev_mask.shape[1], prev_mask.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            xor = cv2.absdiff(cur_mask, prev_mask)
            change_pixels = int((xor > 0).sum())
            delta_powder = powder_count - int((prev_mask > 0).sum())

            xor_big = cv2.resize(xor, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
            xor_heat = colorize_heat(xor_big)
            diff_path = str(DIFF_DIR / f"powder_change_{prev_idx:05d}_{idx:05d}.png")
            cv2.imwrite(diff_path, xor_heat)

        rows.append({
            "frame": cp.name,
            "frame_idx": idx,
            "gray_heat": str(heat_path),
            "powder_L_min": L_MIN,
            "powder_S_max": S_MAX,
            "powder_a_dev": A_DEV,
            "powder_b_dev_pos": B_DEV_POS,
            "powder_b_dev_neg": B_DEV_NEG,
            "powder_count": powder_count,
            "powder_ratio": round(powder_ratio, 6),
            "prev_frame": prev_name or "",
            "delta_powder": delta_powder,
            "change_pixels": change_pixels,
            "powder_change_heatmap": diff_path,
        })

        prev_mask, prev_idx, prev_name = pmask, idx, cp.name

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)

    print(json.dumps({
        "done": True,
        "band_crops_dir": str(BAND_CROPS_DIR),
        "gray_heat_dir": str(HEAT_DIR),
        "diff_heat_dir": str(DIFF_DIR),
        "csv": str(CSV_OUT),
        "frames": len(rows),
        "gating": {
            "L_MIN": L_MIN, "S_MAX": S_MAX,
            "A_DEV": A_DEV, "B_DEV_POS": B_DEV_POS, "B_DEV_NEG": B_DEV_NEG,
            "morph": {"open_k": MORPH_OPEN_K, "close_k": MORPH_CLOSE_K, "min_blob_area": MIN_BLOB_AREA}
        }
    }, indent=2))

if __name__ == "__main__":
    main()
