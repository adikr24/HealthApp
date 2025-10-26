## use step 6 as that's the same and calibrated for the setup ##

from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --------- Paths ---------
ROOT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/greek_yogurt/")
GRAY_HEAT_DIR = ROOT / "gray_heat"      # expects gray_heat_#####.png
CSV_OUT       = ROOT / "gray_heat_volume_calibrated.csv"

# --------- Calibration ---------
TOTAL_MASS_GRAMS = 680.0    # <-- given: total yogurt poured over the whole workflow
grams_per_unit = 5.025234024206276e-07;
# --------- Noise controls (MVP; tweak as needed) ---------
GAUSS_BLUR_KSIZE = 3        # 0/None to disable; else odd int (3 or 5). Blurs ΔL to reduce sparkle
DELTA_THR        = 3        # ignore tiny ΔL magnitudes (0..255). 3~5 is reasonable for 8-bit L
RESIZE_TO_PREV   = True     # if shapes differ, resize current frame to previous before differencing

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
    """
    Convert the blue->red heatmap back to a single-channel intensity (uint8 [0..255]).
    The BGR->GRAY conversion is monotonic with respect to the original luminance used to build the heatmap.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray

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

    total_inflow_units = 0.0
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

            # Make shapes match for fair differencing
            if RESIZE_TO_PREV and cur.shape != ref.shape:
                cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Compute ΔL (signed int16 to avoid wraparound)
            diff = cur.astype(np.int16) - ref.astype(np.int16)

            # Optional small Gaussian blur on ΔL to knock out sparkle
            if GAUSS_BLUR_KSIZE and GAUSS_BLUR_KSIZE > 0 and GAUSS_BLUR_KSIZE % 2 == 1:
                diff = cv2.GaussianBlur(diff, (GAUSS_BLUR_KSIZE, GAUSS_BLUR_KSIZE), 0)

            # Ignore tiny magnitudes (noise)
            if DELTA_THR and DELTA_THR > 0:
                mask_small = (diff > -DELTA_THR) & (diff < DELTA_THR)
                diff[mask_small] = 0

            # Sum of positive and negative ΔL magnitudes
            # Positive = inflow (brightening), Negative = outflow (darkening)
            pos_units = float(diff[diff > 0].sum())
            neg_units = float((-diff[diff < 0]).sum())

            cum_inflow_units += pos_units
            total_inflow_units = cum_inflow_units  # keep updated; final value after loop

        # store row (grams to be filled after we know scale)
        rows.append({
            "frame": fp.name,
            "frame_idx": idx,
            "mean_L": mean_L,
            "pos_delta_units": pos_units,
            "neg_delta_units": neg_units,
            "cum_inflow_units": cum_inflow_units
        })

        prev_gray = gray
        prev_idx = idx

    # ----- Calibration: units -> grams -----
    if total_inflow_units <= 0:
        grams_per_unit = 0.0
    else:
        grams_per_unit = TOTAL_MASS_GRAMS / total_inflow_units

    # Add calibrated grams columns
    for r in rows:
        r["pos_delta_g"]    = r["pos_delta_units"] * grams_per_unit
        r["cum_inflow_g"]   = r["cum_inflow_units"] * grams_per_unit
        r["grams_per_unit"] = grams_per_unit

    # Save CSV
    df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    df.to_csv(CSV_OUT, index=False)

    # Summary
    summary = {
        "done": True,
        "frames": int(len(df)),
        "total_inflow_units": float(total_inflow_units),
        "calibration_total_g": float(TOTAL_MASS_GRAMS),
        "grams_per_unit": float(grams_per_unit),
        "final_cum_inflow_g": float(df["cum_inflow_g"].iloc[-1] if len(df) else 0.0),
        "csv": str(CSV_OUT)
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
