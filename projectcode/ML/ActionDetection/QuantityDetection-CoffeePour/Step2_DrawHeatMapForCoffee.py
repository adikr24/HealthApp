## Step 2 --> Detect pixel-level changes and downward motion within ROIs, draw heatmaps and arrows. ##

import os, csv, json, glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ========= Paths =========
OUTPUT_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/coffee_vol_heat_and_arrows"
FRAMES_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/clean_frames"
CSV_ROI    = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/flow_rois.csv"

OUT_DIR    = os.path.join(OUTPUT_DIR, "roi_pixel_metrics")
HEAT_DIR   = os.path.join(OUT_DIR, "heat_frames")
ARW_DIR    = os.path.join(OUT_DIR, "arrow_overlays")
os.makedirs(HEAT_DIR, exist_ok=True)
os.makedirs(ARW_DIR, exist_ok=True)

# ========= Frame window (inclusive) =========
START_IDX = 50
END_IDX   = 74  # loop uses range(start, end) -> last pair is (73,74) and saves heat_00074.png

# ========= Coffee-friendly knobs (more sensitive) =========
POLARITY            = "both"  # consider both darkening (coffee) and any bright splash
DIFF_THR_BRIGHT_HI  = 12      # HI threshold for brightening
DIFF_THR_DARK_HI    = 8       # HI threshold for darkening (lower = more sensitive)
DIFF_THR_BRIGHT_LO  = 6       # LO threshold for hysteresis grow (bright)
DIFF_THR_DARK_LO    = 4       # LO threshold for hysteresis grow (dark)

# Optical flow gates (softer; good for slow, thin streams)
MAG_THR    = 0.9
VY_THR     = 0.4
VERT_RATIO = 2.0

ARROW_STEP   = 8
ARROW_SCALE  = 2
OUTPUT_SCALE = 4.0
SAVE_ONLY_IF_CHANGE = False

# Trim a small top band to reduce rim glare—but not too much
TOP_CROP_FRAC = 0.05

# Local contrast + gamma brighten
USE_CLAHE = True
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None

def _gamma_LUT(gamma=0.7):
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, gamma)
    return (np.clip(y*255, 0, 255)).astype(np.uint8)
_GAMMA = _gamma_LUT(0.7)

def frame_name(idx):
    return f"frame_{idx:05d}.jpg"

def load_roi_csv(csv_path):
    roi = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing ROI CSV: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fn = r.get("filename", "")
            if not fn:
                continue
            x1 = r.get("roi_x1", ""); y1 = r.get("roi_y1", "")
            x2 = r.get("roi_x2", ""); y2 = r.get("roi_y2", "")
            if (x1 == "" or y1 == "" or x2 == "" or y2 == ""):
                roi[fn] = None
            else:
                roi[fn] = (int(float(x1)), int(float(y1)),
                           int(float(x2)), int(float(y2)))
    return roi

def list_frame_indices(frames_dir):
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    idxs = []
    for p in paths:
        try:
            stem = Path(p).stem
            idx = int(stem.split("_")[1])
            idxs.append(idx)
        except Exception:
            pass
    return sorted(idxs)

def build_blue_to_red_colormap():
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        b = int(round(255 * (1.0 - t)))
        g = 0
        r = int(round(255 * t))
        lut[i, 0, :] = (b, g, r)  # BGR
    return lut
BLUE2RED_LUT = build_blue_to_red_colormap()

def apply_blue2red(gray_0to255):
    gray_u8 = np.clip(gray_0to255, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), BLUE2RED_LUT)

def _to_gray(img_bgr):
    # More robust gray for dark pours: Lab L + CLAHE + gamma brighten
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    if _CLAHE is not None:
        L = _CLAHE.apply(L)
    L = cv2.LUT(L, _GAMMA)
    return L

def _hysteresis_mask(diff_pos_u16, thr_hi, thr_lo):
    """Seed at HI, grow into LO via 3x3 dilations (2 iters)."""
    seeds = (diff_pos_u16 >= thr_hi).astype(np.uint8) * 255
    low   = (diff_pos_u16 >= thr_lo).astype(np.uint8) * 255
    if seeds.max() == 0:
        return np.zeros_like(seeds, dtype=np.uint8)
    grown = seeds.copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    for _ in range(2):
        dil = cv2.dilate(grown, k, iterations=1)
        grown = cv2.bitwise_and(dil, low)
        grown = cv2.bitwise_or(grown, seeds)
    # Connect thin vertical wisps a touch
    grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_RECT, (1,3)), 1)
    return grown

def analyze_pair_roi(A_roi, B_roi):
    """
    Polarity-aware change map (with hysteresis grow) + downward-motion mask within ROI.
    Returns:
      heat_u8  - 0..255 intensity image (for viz)
      downmask - boolean mask (stream-like motion)
      fx, fy   - flow components (float32)
    """
    visA = A_roi.copy()
    visB = B_roi.copy()
    if TOP_CROP_FRAC > 0:
        cut = int(TOP_CROP_FRAC * visA.shape[0])
        visA[:cut, :] = 0
        visB[:cut, :] = 0

    gA = _to_gray(visA)
    gB = _to_gray(visB)

    # Bright (B - A) and Dark (A - B)
    d_bright = (gB.astype(np.int16) - gA.astype(np.int16))
    d_dark   = (gA.astype(np.int16) - gB.astype(np.int16))
    pb = np.clip(d_bright, 0, None).astype(np.uint16)
    pd = np.clip(d_dark,   0, None).astype(np.uint16)

    if POLARITY == "bright":
        seeds = _hysteresis_mask(pb, DIFF_THR_BRIGHT_HI, DIFF_THR_BRIGHT_LO)
        pos_val = pb
    elif POLARITY == "dark":
        seeds = _hysteresis_mask(pd, DIFF_THR_DARK_HI, DIFF_THR_DARK_LO)
        pos_val = pd
    else:
        mb = _hysteresis_mask(pb, DIFF_THR_BRIGHT_HI, DIFF_THR_BRIGHT_LO)
        md = _hysteresis_mask(pd, DIFF_THR_DARK_HI,   DIFF_THR_DARK_LO)
        seeds = cv2.bitwise_or(mb, md)
        pos_val = np.maximum(pb, pd)

    pos_val = (pos_val * (seeds > 0)).astype(np.float32)

    # Normalize for visualization (per-pair)
    if pos_val.max() > 0:
        heat_u8 = (pos_val / (pos_val.max() + 1e-6) * 255).astype(np.uint8)
    else:
        heat_u8 = np.zeros_like(gB, dtype=np.uint8)

    # Optical flow tuned for small motions
    flow = cv2.calcOpticalFlowFarneback(
        gA, gB, None,
        pyr_scale=0.5, levels=2, winsize=11, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0
    )
    fx, fy = flow[...,0], flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)

    strong   = mag > MAG_THR
    downward = fy  > VY_THR
    vertical = np.abs(fy) >= VERT_RATIO * (np.abs(fx) + 1e-6)
    down_mask = (strong & downward & vertical)

    heat_u8[seeds == 0] = 0
    return heat_u8, down_mask, fx, fy

def draw_arrows_on_image(img_full, roi_box, down_mask, fx, fy):
    rx1, ry1, rx2, ry2 = roi_box
    sub_h, sub_w = ry2 - ry1, rx2 - rx1
    for y in range(ARROW_STEP//2, sub_h, ARROW_STEP):
        for x in range(ARROW_STEP//2, sub_w, ARROW_STEP):
            if down_mask[y, x]:
                vx = fx[y, x]; vy = fy[y, x]
                pt1 = (rx1 + x, ry1 + y)
                pt2 = (rx1 + int(x + ARROW_SCALE * vx), ry1 + int(y + ARROW_SCALE * vy))
                cv2.arrowedLine(img_full, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

def main():
    roi_map = load_roi_csv(CSV_ROI)

    all_idxs = list_frame_indices(FRAMES_DIR)
    if not all_idxs:
        print(f"No frames found under {FRAMES_DIR}")
        return
    start = all_idxs[0] if START_IDX is None else START_IDX
    end   = all_idxs[-1] if END_IDX   is None else END_IDX

    rows = []

    for k in range(start, end):
        fnA = frame_name(k)
        fnB = frame_name(k+1)
        fpA = os.path.join(FRAMES_DIR, fnA)
        fpB = os.path.join(FRAMES_DIR, fnB)

        if not (os.path.exists(fpA) and os.path.exists(fpB)):
            continue

        roiA = roi_map.get(fnA, None)
        roiB = roi_map.get(fnB, None)
        if (roiA is None) or (roiB is None):
            continue

        A = cv2.imread(fpA); B = cv2.imread(fpB)
        if A is None or B is None:
            continue
        H, W = A.shape[:2]

        # Intersect consecutive ROIs (small pad)
        ax1, ay1, ax2, ay2 = roiA
        bx1, by1, bx2, by2 = roiB
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        PAD = 2
        rx1 = max(0, ix1 - PAD); ry1 = max(0, iy1 - PAD)
        rx2 = min(W, ix2 + PAD);  ry2 = min(H, iy2 + PAD)
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        A_roi = A[ry1:ry2, rx1:rx2]
        B_roi = B[ry1:ry2, rx1:rx2]

        heat_u8, down_mask, fx, fy = analyze_pair_roi(A_roi, B_roi)

        if SAVE_ONLY_IF_CHANGE and heat_u8.max() == 0 and not down_mask.any():
            continue

        # Heat (ROI-only), scaled
        heat_roi_bgr = apply_blue2red(heat_u8)
        heat_vis = cv2.resize(
            heat_roi_bgr, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        heat_path = os.path.join(HEAT_DIR, f"heat_{k+1:05d}.png")
        cv2.imwrite(heat_path, heat_vis)

        # Arrow overlay (ROI-only), scaled
        arrow_roi = B[ry1:ry2, rx1:rx2].copy()
        draw_arrows_on_image(
            arrow_roi, (0, 0, arrow_roi.shape[1], arrow_roi.shape[0]),
            down_mask, fx, fy
        )
        arrow_vis = cv2.resize(
            arrow_roi, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        arrows_path = os.path.join(ARW_DIR, f"arrows_{k+1:05d}.png")
        cv2.imwrite(arrows_path, arrow_vis)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, "viz_summary.csv"), index=False)

    meta = {
        "frames_dir": FRAMES_DIR,
        "roi_csv": CSV_ROI,
        "heat_dir": HEAT_DIR,
        "arrows_dir": ARW_DIR,
        "polarity": POLARITY,
        "thr_bright_hi": DIFF_THR_BRIGHT_HI,
        "thr_dark_hi": DIFF_THR_DARK_HI,
        "range_used": [start, end]
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
