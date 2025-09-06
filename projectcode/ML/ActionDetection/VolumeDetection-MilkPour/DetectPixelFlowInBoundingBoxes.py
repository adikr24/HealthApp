## Step 2 --> ## Detect pixel-level changes and downward motion within ROIs, draw heatmaps and arrows. ##

import os, csv, json, glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ========= Paths =========
# Use the SAME output dir your detector wrote annotated frames + flow_rois.csv into
OUTPUT_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_vol_heat_and_arrows"
FRAMES_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_volume_output/clean_frames/"                          # annotated frames with boxes
CSV_ROI    = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_volume_output/flow_rois.csv" # os.path.join(OUTPUT_DIR, "flow_rois.csv")

OUT_DIR    = os.path.join(OUTPUT_DIR, "roi_pixel_metrics")
HEAT_DIR   = os.path.join(OUT_DIR, "heat_frames")
ARW_DIR    = os.path.join(OUT_DIR, "arrow_overlays")
os.makedirs(HEAT_DIR, exist_ok=True)
os.makedirs(ARW_DIR, exist_ok=True)

# ========= Frame window (inclusive) =========
# Leave None to auto-detect full range from available frames; or set ints.
START_IDX = None
END_IDX   = None

# ========= Thresholds / knobs =========
# Pixel-difference (B - A) gate for “new bright (milk-like)” pixels
DIFF_THR   = 15        # gray delta
# Optical flow (Farnebäck) gates for “strong downward vertical motion”
MAG_THR    = 1.2       # min flow magnitude (px/frame)
VY_THR     = 0.6       # min positive vy
VERT_RATIO = 2.5       # |vy| >= VERT_RATIO * |vx|

# Visualization knobs
ARROW_STEP  = 8        # draw one arrow every N pixels
ARROW_SCALE = 2        # length multiplier
HEAT_ALPHA  = 0.55     # blend heatmap into blue canvas
SAVE_ONLY_IF_CHANGE = False  # if True, only save frames with any positive diff in ROI
OUTPUT_SCALE = 4.0   # 2.0 = double size, 3.0 = triple size

# Optional: ignore the very top of ROI (rim glare)
TOP_CROP_FRAC = 0.10   # set to 0.0 to disable

def frame_name(idx): 
    return f"frame_{idx:05d}.jpg"

def load_roi_csv(csv_path):
    """Return dict: filename -> (rx1,ry1,rx2,ry2) or None if missing."""
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
            stem = Path(p).stem  # "frame_00299"
            idx = int(stem.split("_")[1])
            idxs.append(idx)
        except Exception:
            pass
    return sorted(idxs)

def build_blue_to_red_colormap():
    """
    Custom 0..255 -> BGR gradient:
    0 = deep blue, 255 = red (no greens in between).
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        # Simple two-stop gradient: blue -> red
        # Blue channel decreases, Red increases, Green stays 0
        b = int(round(255 * (1.0 - t)))
        g = 0
        r = int(round(255 * t))
        lut[i, 0, :] = (b, g, r)  # BGR
    return lut

BLUE2RED_LUT = build_blue_to_red_colormap()

def apply_blue2red(gray_0to255):
    """Map single-channel 0..255 -> BGR using custom blue->red LUT."""
    gray_u8 = np.clip(gray_0to255, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), BLUE2RED_LUT)

def analyze_pair_roi(A_roi, B_roi):
    """
    Compute positive-diff heat + downward optical flow mask within ROI.
    Returns:
      pos_diff_u8  - normalized 0..255 intensities for positives (0 elsewhere)
      down_mask    - boolean mask (stream-like motion)
      fx, fy       - flow components (float32)
    """
    visA = A_roi.copy()
    visB = B_roi.copy()
    if TOP_CROP_FRAC > 0:
        cut = int(TOP_CROP_FRAC * visA.shape[0])
        visA[:cut, :] = 0
        visB[:cut, :] = 0

    gA = cv2.cvtColor(visA, cv2.COLOR_BGR2GRAY)
    gB = cv2.cvtColor(visB, cv2.COLOR_BGR2GRAY)

    diff = (gB.astype(np.int16) - gA.astype(np.int16))
    pos  = (diff > DIFF_THR).astype(np.uint8)

    # Normalize only the positive part for visualization
    pos_diff = np.clip(diff, 0, None).astype(np.float32)
    if pos_diff.max() > 0:
        pos_diff_u8 = (pos_diff / (pos_diff.max() + 1e-6) * 255).astype(np.uint8)
    else:
        pos_diff_u8 = np.zeros_like(gB, dtype=np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        gA, gB, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    fx, fy = flow[...,0], flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)

    strong   = mag > MAG_THR
    downward = fy > VY_THR
    vertical = np.abs(fy) >= VERT_RATIO * (np.abs(fx) + 1e-6)
    down_mask = (strong & downward & vertical)

    # Zero-out non-positives in pos_diff_u8 so heat only shows new-bright pixels
    pos_diff_u8[ pos == 0 ] = 0

    return pos_diff_u8, down_mask, fx, fy

def draw_arrows_on_image(img_full, roi_box, down_mask, fx, fy):
    """Overlay green arrows (only where down_mask==True) into img_full within ROI."""
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
    # Load ROIs
    roi_map = load_roi_csv(CSV_ROI)

    # Determine frame range
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
            # skip pairs without a valid ROI on both frames
            continue

        A = cv2.imread(fpA); B = cv2.imread(fpB)
        if A is None or B is None:
            continue
        H, W = A.shape[:2]

        # Lock region by intersecting consecutive frame ROIs (then small pad)
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

        pos_diff_u8, down_mask, fx, fy = analyze_pair_roi(A_roi, B_roi)

        # Decide whether to save based on change presence
        if SAVE_ONLY_IF_CHANGE and pos_diff_u8.max() == 0 and not down_mask.any():
            continue

        # ----- Build and save HEAT image (full frame with ROI pasted) -----
        # Map positive diffs to blue->red colors (red = stronger change)
        heat_roi_bgr = apply_blue2red(pos_diff_u8)
        # Blend heat into a dark blue canvas for clarity, then paste into full frame canvas
        # (Or paste directly—here we paste the colored ROI into a black full-frame canvas)
        heat_full = np.zeros_like(B)
        heat_full[ry1:ry2, rx1:rx2] = heat_roi_bgr

        heat_path = os.path.join(HEAT_DIR, f"heat_{k+1:05d}.png")

        heat_roi_bgr = apply_blue2red(pos_diff_u8)  # colorize positive diffs within the ROI only
        heat_vis = cv2.resize(
            heat_roi_bgr,
            None,
            fx=OUTPUT_SCALE,
            fy=OUTPUT_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(heat_path, heat_vis)

        # ----- ROI-only ARROWS (scaled) -----
        arrow_roi = B[ry1:ry2, rx1:rx2].copy()   # crop ROI from frame B
        draw_arrows_on_image(
            arrow_roi,
            (0, 0, arrow_roi.shape[1], arrow_roi.shape[0]),  # ROI coords relative to crop
            down_mask, fx, fy
        )

        arrow_vis = cv2.resize(
            arrow_roi,
            None,
            fx=OUTPUT_SCALE,
            fy=OUTPUT_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        arrows_path = os.path.join(ARW_DIR, f"arrows_{k+1:05d}.png")
        cv2.imwrite(arrows_path, arrow_vis)

    # Optional: write a small summary CSV next to outputs
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, "viz_summary.csv"), index=False)

    meta = {
        "frames_dir": FRAMES_DIR,
        "roi_csv": CSV_ROI,
        "heat_dir": HEAT_DIR,
        "arrows_dir": ARW_DIR,
        "range_used": [start, end]
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
