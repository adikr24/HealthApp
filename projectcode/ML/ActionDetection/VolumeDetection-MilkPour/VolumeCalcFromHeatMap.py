## Step 4 --> Estimate volume from heatmap scores ##

import os, csv, glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# ====== INPUTS ======
HEAT_DIR   = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_vol_heat_and_arrows/roi_pixel_metrics/rim_filtered_keep_all_flow"
START_IDX  = 0
END_IDX    = 300           # inclusive
FNAME_FMT  = "heat_{:05d}.png"

# Ignore very dim reds (noise) and an optional top crop to reduce rim glare
RED_THR        = 8         # 0..255 threshold on red channel
TOP_CROP_FRAC  = 0.10      # crop top X% of the image (0.0 to disable)

# Convert score -> mL after one-time calibration (set this later)
CAL_ML_PER_SCORE = 0.0006  # placeholder; set = known_ml / sum_scores_from_cal_clip

# ====== OUTPUTS ======
OUT_DIR   = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_vol_heat_and_arrows/roi_pixel_metrics/remove_glare"  # roi_pixel_metrics
CSV_PATH  = os.path.join(OUT_DIR, "heat_only_flow_per_frame.csv")

def load_heat(idx):
    fp = os.path.join(HEAT_DIR, FNAME_FMT.format(idx))
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    return fp, im

def frame_score_from_heat(heat_bgr, top_crop_frac=0.0, red_thr=0):
    """
    Heat images were created from a blue->red LUT where red ~ intensity, blue ~ background.
    So we can simply use the red channel as the per-pixel strength in [0..255].
    """
    if heat_bgr is None:
        return 0.0, 0  # score, pixels_used

    H, W = heat_bgr.shape[:2]
    y0 = int(round(H * top_crop_frac))
    roi = heat_bgr[y0:, :, :] if y0 < H else heat_bgr

    # Red channel (BGR order)
    red = roi[..., 2].astype(np.float32)

    # Mask: keep only red>thr to ignore blue background + tiny flickers
    if red_thr > 0:
        mask = red > red_thr
        red = red * mask

    # Normalize to [0..1] so score is roughly “area × intensity”
    score = float(red.sum() / 255.0)
    used  = int((red > 0).sum())
    return score, used

def main():
    rows = []
    cum_score = 0.0
    cum_ml    = 0.0

    for idx in range(START_IDX, END_IDX + 1):
        fp, heat = load_heat(idx)
        if heat is None:
            print(f"Missing heat image: {fp}")
            continue

        score, n_pix = frame_score_from_heat(
            heat_bgr=heat,
            top_crop_frac=TOP_CROP_FRAC,
            red_thr=RED_THR
        )
        cum_score += score
        delta_ml = score * CAL_ML_PER_SCORE
        cum_ml   += delta_ml

        rows.append({
            "heat_filename": Path(fp).name,
            "index": idx,
            "red_thr": RED_THR,
            "top_crop_frac": TOP_CROP_FRAC,
            "active_pixels": n_pix,
            "flow_score": score,       # arbitrary units
            "cum_score": cum_score,
            "delta_ml": delta_ml,      # needs calibration constant above
            "cum_ml": cum_ml
        })

    if not rows:
        print("No frames processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved per-frame metrics → {CSV_PATH}")
    print(f"Total frames: {len(rows)}")
    print(f"Total score:  {df['flow_score'].sum():.2f}")
    print(f"Total mL:     {df['delta_ml'].sum():.2f} (using CAL_ML_PER_SCORE={CAL_ML_PER_SCORE})")

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    main()
