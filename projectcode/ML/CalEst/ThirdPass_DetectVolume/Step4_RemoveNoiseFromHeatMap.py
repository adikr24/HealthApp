#!/usr/bin/env python3
# Remove THIN BLOBS (not thick enough). Keep only thick/compact motion (e.g., falling blueberries).
# Input: heatmaps (blue→red) in HEAT_DIR
# Output: filtered images to OUT_DIR with thin blobs removed.

import os, glob
from pathlib import Path
import numpy as np
import cv2

# ---------- Paths ----------
HEAT_DIR = "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/band_debug/diff_heat"
OUT_DIR  = os.path.join(str(Path(HEAT_DIR).parent), "thinblob_stripped")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Knobs ----------
# Color gating (what counts as “red-ish” diff)
RED_THR = 10          # min red channel
RED_DOM = 10          # red must exceed max(G,B) by at least this much

# Thin-blob removal thresholds (default = moderate)
MIN_AREA_PX   = 40    # drop tiny specks
MIN_WIDTH_PX  = 6     # require blob bbox "thickness" (min(w,h)) ≥ this
MIN_RADIUS_PX = 3.5   # require blob distance-transform max radius ≥ this (≈ half-thickness)

# Make it stricter in one switch (raise bars = remove more)
STRICT = False
if STRICT:
    MIN_AREA_PX   = 80
    MIN_WIDTH_PX  = 8
    MIN_RADIUS_PX = 4.5

# ---------- Helpers ----------
def list_inputs():
    files = []
    for ex in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(HEAT_DIR, ex))
    return sorted(files)

def red_mask(img):
    R = img[...,2].astype(np.int16)
    G = img[...,1].astype(np.int16)
    B = img[...,0].astype(np.int16)
    return ((R >= RED_THR) & ((R - np.maximum(G, B)) >= RED_DOM)).astype(np.uint8)

# ---------- Main ----------
def main():
    inputs = list_inputs()
    if not inputs:
        print(f"[WARN] No images in {HEAT_DIR}")
        return

    for fp in inputs:
        im = cv2.imread(fp, cv2.IMREAD_COLOR)
        if im is None:
            continue
        H, W = im.shape[:2]

        rm = red_mask(im)  # 0/1 mask of red-ish pixels

        # Connected components on red mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats(rm, connectivity=8)

        # Build keep/remove masks
        keep_mask = np.zeros((H, W), np.uint8)
        for lab in range(1, num):
            x, y, w, h, area = stats[lab]
            if area < MIN_AREA_PX:
                continue  # too small → drop

            comp = (labels == lab).astype(np.uint8)

            # Geometric "thickness" by bbox
            min_width = min(w, h)

            # True thickness via distance transform (max inscribed radius)
            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
            max_r = float(dt.max())  # ~ half of local thickness in pixels

            # Keep only if blob is thick enough by BOTH criteria
            if (min_width >= MIN_WIDTH_PX) and (max_r >= MIN_RADIUS_PX):
                keep_mask[comp > 0] = 1
            # else: thin/liney → dropped

        # Compose output: remove dropped red pixels back to background blue
        out = im.copy()
        nonred = (rm == 0)
        bg_b = int(np.median(im[...,0][nonred])) if np.any(nonred) else 64
        out[(rm == 1) & (keep_mask == 0)] = (bg_b, 0, 0)

        cv2.imwrite(os.path.join(OUT_DIR, Path(fp).name), out)

    print(f"[DONE] Saved filtered frames to: {OUT_DIR}")
    print(f"Params: STRICT={STRICT}, MIN_AREA_PX={MIN_AREA_PX}, "
          f"MIN_WIDTH_PX={MIN_WIDTH_PX}, MIN_RADIUS_PX={MIN_RADIUS_PX}, "
          f"RED_THR={RED_THR}, RED_DOM={RED_DOM}")

if __name__ == "__main__":
    main()
