#!/usr/bin/env python3
# Pinned-calibration inflow from gray_heat PNGs.
# Uses grams_per_unit learned previously, so no per-run total grams needed.

from __future__ import annotations
import re, json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --------- Paths ---------
ROOT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/greek_yogurt/")
GRAY_HEAT_DIR = ROOT / "gray_heat"      # expects gray_heat_#####.png
CSV_OUT       = ROOT / "gray_heat_volume_calibrated.csv"

# --------- Pinned calibration (from previous run) ---------
GRAMS_PER_UNIT = 5.025234024206276e-07  # g per ΔL unit (keep this constant unless setup changes)

# --------- Noise controls (same MVP knobs) ---------
GAUSS_BLUR_KSIZE = 3   # 0/None to disable; else 3 or 5
DELTA_THR        = 3   # ignore tiny |ΔL| (< DELTA_THR)
RESIZE_TO_PREV   = True

# --------- Helpers ---------
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

def to_gray_from_heat(bgr: np.ndarray) -> np.ndarray:
    # Monotonic mapping back to intensity (uint8 [0..255]) from blue->red heatmap
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# --------- Main ---------
def main():
    files = list(GRAY_HEAT_DIR.glob("gray_heat_*.png"))
    files.sort(key=lambda p: idx_from_fn(p.name))
    if not files:
        print(json.dumps({"error": "no gray_heat PNGs found", "dir": str(GRAY_HEAT_DIR)}, indent=2))
        return

    rows = []
    prev_gray: Optional[np.ndarray] = None
    prev_idx: Optional[int] = None

    cum_inflow_units = 0.0

    for fp in files:
        idx = idx_from_fn(fp.name)
        bgr = imread_any(fp)
        if bgr is None:
            rows.append({"frame": fp.name, "frame_idx": idx, "error": "read_fail"})
            prev_gray = None
            prev_idx = idx
            continue

        gray = to_gray_from_heat(bgr)  # uint8 [0..255]
        mean_L = float(gray.mean())

        pos_units = 0.0
        neg_units = 0.0

        if prev_gray is not None:
            cur = gray
            ref = prev_gray

            # Match shapes for fair ΔL
            if RESIZE_TO_PREV and cur.shape != ref.shape:
                cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

            # ΔL as signed int16
            diff = cur.astype(np.int16) - ref.astype(np.int16)

            # De-sparkle ΔL (optional)
            if GAUSS_BLUR_KSIZE and GAUSS_BLUR_KSIZE % 2 == 1:
                diff = cv2.GaussianBlur(diff, (GAUSS_BLUR_KSIZE, GAUSS_BLUR_KSIZE), 0)

            # Ignore tiny magnitudes
            if DELTA_THR and DELTA_THR > 0:
                mask_small = (diff > -DELTA_THR) & (diff < DELTA_THR)
                diff[mask_small] = 0

            # Sum positive (inflow) and negative (outflow) magnitudes
            pos_units = float(diff[diff > 0].sum())
            neg_units = float((-diff[diff < 0]).sum())

            cum_inflow_units += pos_units

        # Calibrate to grams with pinned constant
        pos_g  = pos_units * GRAMS_PER_UNIT
        cum_g  = cum_inflow_units * GRAMS_PER_UNIT

        rows.append({
            "frame": fp.name,
            "frame_idx": idx,
            "mean_L": mean_L,
            "pos_delta_units": pos_units,
            "neg_delta_units": neg_units,
            "cum_inflow_units": cum_inflow_units,
            "pos_delta_g": pos_g,
            "cum_inflow_g": cum_g,
            "grams_per_unit": GRAMS_PER_UNIT
        })

        prev_gray = gray
        prev_idx = idx

    # Save CSV
    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)

    # Summary
    summary = {
        "done": True,
        "frames": int(len(df)),
        "final_cum_inflow_units": float(df["cum_inflow_units"].iloc[-1] if len(df) else 0.0),
        "grams_per_unit": float(GRAMS_PER_UNIT),
        "final_cum_inflow_g": float(df["cum_inflow_g"].iloc[-1] if len(df) else 0.0),
        "csv": str(CSV_OUT)
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
