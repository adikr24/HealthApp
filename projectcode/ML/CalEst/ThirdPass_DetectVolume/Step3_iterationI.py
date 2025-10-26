#!/usr/bin/env python3
# Count blueberry-like blobs per heatmap; print "<filename>: <count>" and cumulative total

import os, glob
from pathlib import Path
import numpy as np
import cv2
from math import pi

HEAT_DIR = "/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/band_debug/diff_heat"

# ------- Knobs -------
RED_THR    = 32
RED_DOM    = 20
BLUR_K     = 0

MIN_AREA   = 150
THICK_MIN  = 5.2
CIRC_MIN   = 2.4
AR_MAX     = 7.5

DT_PEAK_MIN   = 6.0
PEAK_SEP      = 16
BORDER_MIN_R  = 6.0

ASSUMED_BERRY_R = 6.5
AREA_PER_BERRY  = pi * (ASSUMED_BERRY_R ** 2)

# ------- Helpers -------
def list_imgs(folder):
    files = []
    for ex in ("*.png", "*.jpg", "*.jpeg"):
        files += glob.glob(os.path.join(folder, ex))
    return sorted(files)

def red_mask(img):
    if BLUR_K and BLUR_K > 1:
        k = BLUR_K | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    R = img[...,2].astype(np.int16)
    G = img[...,1].astype(np.int16)
    B = img[...,0].astype(np.int16)
    return ((R >= RED_THR) & ((R - np.maximum(G, B)) >= RED_DOM)).astype(np.uint8)

def circularity(binmask):
    cnts,_ = cv2.findContours((binmask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return 0.0
    c = max(cnts, key=cv2.contourArea)
    A = float(cv2.contourArea(c))
    P = float(cv2.arcLength(c, True))
    if P < 1e-6: return 0.0
    return float((4.0*pi*A)/(P*P))

def dt_peaks(dt, thresh, min_dist):
    """Distance-transform peak finder with safe suppression using cv2.circle."""
    if dt.size == 0: return []
    cand = (dt >= thresh)
    if not cand.any(): return []
    k = 2*int(round(min_dist)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(dt, kernel)
    local = (dt == dil) & cand
    ys, xs = np.where(local)
    if ys.size == 0: return []
    vals = dt[ys, xs]
    order = np.argsort(-vals)
    ys, xs, vals = ys[order], xs[order], vals[order]
    blocked = np.zeros(dt.shape, np.uint8)
    rad = max(1, int(round(min_dist)))
    taken = []
    for y, x, v in zip(ys, xs, vals):
        if blocked[y, x]:
            continue
        taken.append((int(y), int(x), float(v)))
        cv2.circle(blocked, (int(x), int(y)), rad, 1, thickness=-1, lineType=cv2.LINE_AA)
    return taken

# ------- Main -------
def main():
    files = list_imgs(HEAT_DIR)
    if not files:
        print(f"[WARN] no images in {HEAT_DIR}")
        return

    total_all = 0  # cumulative total counter

    for fp in files:
        im = cv2.imread(fp, cv2.IMREAD_COLOR)
        if im is None:
            continue

        mask = red_mask(im)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total = 0

        for lab in range(1, num):
            x,y,w,h,area = stats[lab]
            if area < MIN_AREA:
                continue
            comp = (labels == lab).astype(np.uint8)
            ar = max(w,h) / max(1.0, min(w,h))
            circ = circularity(comp)
            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3).astype(np.float32)
            max_r = float(dt.max())

            if max_r < THICK_MIN and ar > 2.0 and circ < 0.2:
                continue
            if (circ < CIRC_MIN) and (max_r < THICK_MIN*1.2) and (ar > AR_MAX):
                continue

            peaks = dt_peaks(dt, DT_PEAK_MIN, PEAK_SEP)
            peaks = [(py, px, r) for (py, px, r) in peaks if r >= BORDER_MIN_R]

            max_allowed = max(1, int(round(area / AREA_PER_BERRY)))
            if len(peaks) > max_allowed:
                peaks = sorted(peaks, key=lambda t: t[2], reverse=True)[:max_allowed]

            if not peaks and (max_r >= THICK_MIN) and (circ >= CIRC_MIN):
                peaks = [(y + h//2, x + w//2, max_r)]

            total += len(peaks)

        total_all += total
        print(f"{Path(fp).name}: {total}")

    # ---- Print cumulative summary ----
    print(f"\n[SUMMARY] cumulative blueberry count: {total_all}")

if __name__ == "__main__":
    main()
