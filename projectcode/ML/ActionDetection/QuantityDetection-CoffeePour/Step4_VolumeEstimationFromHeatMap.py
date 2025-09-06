# Step 4 --> Estimate volume/weight from heatmap scores (gate-band + scale fix + grams)

import os, glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# ====== INPUTS ======
HEAT_DIR   = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/coffee_vol_heat_and_arrows/roi_pixel_metrics/rim_filtered_keep_all_flow"
START_IDX  = 0
END_IDX    = 300           # inclusive
FNAME_FMT  = "heat_{:05d}.png"

# The Step-2 heat images were saved at OUTPUT_SCALE=4.0 (enlarged viz)
OUTPUT_SCALE = 4.0
SCALE_CORRECTION = 1.0 / (OUTPUT_SCALE ** 2)   # undo area inflation

# Thresholds
RED_THR        = 12         # ignore very dim red (noise); try 8..15
TOP_CROP_FRAC  = 0.00       # gate handles rim glare; keep 0 unless needed

# ---- Gate band (count only crossing pixels) ----
GATE_OFFSET_FRAC = 0.08     # band center ~8% down from top of image
GATE_BAND_FRAC   = 0.06     # band thickness ~6% of image height

# Convert score -> mL after one-time calibration (set this for YOUR pipeline)
CAL_ML_PER_SCORE = 0.0008   # recompute after changes!

# ---- mL -> grams (density) ----
# For liquids (brewed coffee/water-like): ~1.0
# For instant coffee powder bulk: ~0.32–0.40 (set from your calibration)
DENSITY_G_PER_ML = 1.0

# ====== OUTPUTS ======
OUT_DIR   = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/coffee_vol_heat_and_arrows/roi_pixel_metrics/remove_glare"
CSV_PATH  = os.path.join(OUT_DIR, "heat_only_flow_per_frame.csv")

def load_heat(idx):
    fp = os.path.join(HEAT_DIR, FNAME_FMT.format(idx))
    im = cv2.imread(fp, cv2.IMREAD_COLOR)
    return fp, im

def frame_score_from_heat(heat_bgr,
                          red_thr=0,
                          top_crop_frac=0.0,
                          gate_offset_frac=0.08,
                          gate_band_frac=0.06):
    """
    Sum red-channel intensities ONLY within a thin gate band under the rim.
    Apply scale correction to undo Step-2's visualization upscaling.
    Returns (score, n_active_pixels, gate_y1, gate_y2).
    """
    if heat_bgr is None:
        return 0.0, 0, 0, 0

    H, W = heat_bgr.shape[:2]

    # Optional top crop (usually 0 because gate handles rim glare)
    y0 = int(round(H * top_crop_frac))
    if y0 >= H:
        return 0.0, 0, 0, 0
    img = heat_bgr[y0:, :, :]
    HH = img.shape[0]

    # Gate band in the cropped image coordinates
    gate_center = int(round(gate_offset_frac * HH))
    gate_h      = int(max(1, round(gate_band_frac * HH)))
    gy1 = max(0, gate_center - gate_h // 2)
    gy2 = min(HH, gate_center + (gate_h - gate_h // 2))

    gate_roi = img[gy1:gy2, :, :]
    red = gate_roi[..., 2].astype(np.float32)

    if red_thr > 0:
        mask = red > red_thr
        red = red * mask

    raw_sum = red.sum() / 255.0

    # Undo Step-2's OUTPUT_SCALE enlargement
    score = float(raw_sum * SCALE_CORRECTION)

    used  = int((red > 0).sum())
    # return gate y in full image coordinates for debugging
    return score, used, y0 + gy1, y0 + gy2

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    have = {int(Path(p).stem.split("_")[1]) for p in glob.glob(os.path.join(HEAT_DIR, "heat_*.png")) if "_" in Path(p).stem}
    if not have:
        print(f"No heat frames found in {HEAT_DIR}")
        return

    rows = []
    cum_score = 0.0
    cum_ml    = 0.0
    cum_g     = 0.0

    lo = START_IDX if START_IDX is not None else (min(have) if have else 0)
    hi = END_IDX   if END_IDX   is not None else (max(have) if have else 0)

    for idx in range(lo, hi + 1):
        if idx not in have:
            continue

        fp, heat = load_heat(idx)
        score, n_pix, gy1, gy2 = frame_score_from_heat(
            heat_bgr=heat,
            red_thr=RED_THR,
            top_crop_frac=TOP_CROP_FRAC,
            gate_offset_frac=GATE_OFFSET_FRAC,
            gate_band_frac=GATE_BAND_FRAC
        )

        delta_ml = score * CAL_ML_PER_SCORE
        delta_g  = delta_ml * DENSITY_G_PER_ML

        cum_score += score
        cum_ml    += delta_ml
        cum_g     += delta_g

        rows.append({
            "heat_filename": Path(fp).name,
            "index": idx,
            "red_thr": RED_THR,
            "top_crop_frac": TOP_CROP_FRAC,
            "gate_offset_frac": GATE_OFFSET_FRAC,
            "gate_band_frac": GATE_BAND_FRAC,
            "active_pixels": n_pix,
            "flow_score": score,       # area × intensity within gate (scale-corrected)
            "delta_ml": delta_ml,
            "cum_ml": cum_ml,
            "delta_g": delta_g,        # grams for this frame
            "cum_g": cum_g,            # cumulative grams
            "gate_y1": gy1,
            "gate_y2": gy2
        })

    if not rows:
        print("No frames processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved per-frame metrics → {CSV_PATH}")
    print(f"Total frames: {len(rows)}")
    print(f"Total score (gate, scale-corrected): {df['flow_score'].sum():.2f}")
    print(f"Total mL:    {df['delta_ml'].sum():.2f}  (CAL_ML_PER_SCORE={CAL_ML_PER_SCORE})")
    print(f"Total grams: {df['delta_g'].sum():.2f}  (DENSITY_G_PER_ML={DENSITY_G_PER_ML})")

if __name__ == "__main__":
    main()
