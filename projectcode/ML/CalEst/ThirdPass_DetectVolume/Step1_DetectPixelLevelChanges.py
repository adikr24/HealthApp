#!/usr/bin/env python3
"""
Third Pass — Volume detection visualizations for a single OOI action sequence
- Selects greek_yogurt (2nd action sequence) from vol_detection_reference.csv
- Slices ROI CSV to the frame window
- Generates per-pair ROI heatmaps (blue→red) and arrow overlays
- Copies the exact frames into clean_frames/
"""

import os, csv, json, glob, shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ========= Inputs (SecondPass metadata) =========
VOL_REF_CSV = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/ActivityFrameSegmentation/vol_detection_reference.csv")
ROI_MASTER_CSV_DEFAULT = VOL_REF_CSV.with_name("flow_rois.csv")

# Source frames (full extracted frames for the video)
FRAMES_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIBoundingBoxes")

# ========= Output root (ThirdPass) =========
BASE_OUT = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume")

# ========= Coffee/yogurt-friendly knobs =========
POLARITY            = "bright"  # "bright" (default for greek yogurt), "dark", or "both"
DIFF_THR_BRIGHT_HI  = 12
DIFF_THR_DARK_HI    = 8
DIFF_THR_BRIGHT_LO  = 6
DIFF_THR_DARK_LO    = 4

MAG_THR    = 0.9
VY_THR     = 0.4
VERT_RATIO = 2.0

ARROW_STEP   = 8
ARROW_SCALE  = 2
OUTPUT_SCALE = 4.0
SAVE_ONLY_IF_CHANGE = False

TOP_CROP_FRAC = 0.05

USE_CLAHE = True
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None

def _gamma_LUT(gamma=0.7):
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, gamma)
    return (np.clip(y*255, 0, 255)).astype(np.uint8)
_GAMMA = _gamma_LUT(0.7)

def frame_name(idx):
    return f"frame_{idx:05d}.jpg"

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
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    if _CLAHE is not None:
        L = _CLAHE.apply(L)
    L = cv2.LUT(L, _GAMMA)
    return L

def _hysteresis_mask(diff_pos_u16, thr_hi, thr_lo):
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
    grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_RECT, (1,3)), 1)
    return grown

def analyze_pair_roi(A_roi, B_roi):
    visA = A_roi.copy()
    visB = B_roi.copy()
    if TOP_CROP_FRAC > 0:
        cut = int(TOP_CROP_FRAC * visA.shape[0])
        visA[:cut, :] = 0
        visB[:cut, :] = 0

    gA = _to_gray(visA)
    gB = _to_gray(visB)

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
    heat_u8 = (pos_val / (pos_val.max() + 1e-6) * 255).astype(np.uint8) if pos_val.max() > 0 else np.zeros_like(gB, dtype=np.uint8)

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

def load_reference_and_pick_sequence(vol_ref_csv: Path):
    if not vol_ref_csv.exists():
        raise FileNotFoundError(f"Missing vol_detection_reference.csv: {vol_ref_csv}")
    df = pd.read_csv(vol_ref_csv)
    # Normalize headers we rely on
    must_cols = {"start_idx","end_idx","start_frame","end_frame","classes"}
    if not must_cols.issubset(set(df.columns)):
        raise ValueError(f"{vol_ref_csv} missing required columns (needs at least {must_cols})")

    # Try to use OOI if present
    if "OOI" in df.columns:
        # prefer greek_yogurt row
        mask = df["OOI"].astype(str).str.lower() == "greek_yogurt"
        if mask.any():
            row = df[mask].iloc[0]
        else:
            # fallback: second row by input order
            row = df.iloc[1] if len(df) >= 2 else df.iloc[0]
        ooi_name = str(row.get("OOI", "greek_yogurt")).strip()
    else:
        # no OOI column: fallback to 2nd action row
        row = df.iloc[1] if len(df) >= 2 else df.iloc[0]
        ooi_name = "greek_yogurt"

    start_idx = int(row["start_idx"])
    end_idx   = int(row["end_idx"])
    # inclusive handling like your coffee script: process pairs (k, k+1) up to end_idx
    return ooi_name, start_idx, end_idx

def load_roi_csv(csv_path: Path):
    roi = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ROI CSV: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        # expected columns: filename, roi_x1, roi_y1, roi_x2, roi_y2
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

def list_frame_indices(frames_dir: Path):
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    idxs = []
    for p in paths:
        try:
            stem = p.stem
            idx = int(stem.split("_")[1])
            idxs.append(idx)
        except Exception:
            pass
    return sorted(idxs)

def slice_and_save_roi_csv(master_csv: Path, out_csv: Path, start_idx: int, end_idx: int):
    if not master_csv.exists():
        raise FileNotFoundError(f"ROI master CSV not found: {master_csv}")
    df = pd.read_csv(master_csv)
    if not {"filename","roi_x1","roi_y1","roi_x2","roi_y2"}.issubset(df.columns):
        raise ValueError(f"{master_csv} must have columns: filename, roi_x1, roi_y1, roi_x2, roi_y2")

    def idx_from_fn(fn):
        try:
            return int(Path(fn).stem.split("_")[1])
        except Exception:
            return -1

    df["__idx"] = df["filename"].astype(str).map(idx_from_fn)
    sliced = df[(df["__idx"] >= start_idx) & (df["__idx"] <= end_idx)].drop(columns="__idx")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out_csv, index=False)
    return out_csv

def main():
    # 1) Pick the action window (greek_yogurt / 2nd sequence)
    ooi_name, START_IDX, END_IDX = load_reference_and_pick_sequence(VOL_REF_CSV)
    ooi_slug = ooi_name.lower().replace(" ", "_")

    # 2) Prepare output dirs for this OOI
    OOI_OUT = BASE_OUT / ooi_slug
    OUT_DIR = OOI_OUT  # keep metadata here
    HEAT_DIR = OOI_OUT / "vol_heat"
    ARW_DIR  = OOI_OUT / "arrows"
    CLEAN_DIR= OOI_OUT / "clean_frames"
    for d in (HEAT_DIR, ARW_DIR, CLEAN_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 3) Make a sliced ROI CSV for this window
    roi_master_csv = ROI_MASTER_CSV_DEFAULT
    ROI_OUT_CSV = OOI_OUT / "flow_rois.csv"
    try:
        slice_and_save_roi_csv(roi_master_csv, ROI_OUT_CSV, START_IDX, END_IDX)
    except Exception as e:
        raise RuntimeError(f"Failed to slice ROI CSV: {e}")

    # 4) Load ROI map (only sliced rows)
    roi_map = load_roi_csv(ROI_OUT_CSV)

    # 5) Build list of frames & copy into clean_frames/
    all_idxs = list_frame_indices(FRAMES_DIR)
    if not all_idxs:
        print(f"No frames found under {FRAMES_DIR}")
        return
    # clamp to existing frames
    start = max(START_IDX, all_idxs[0])
    end   = min(END_IDX,   all_idxs[-1])

    # Copy frames to clean_frames for this window
    for k in range(start, end+1):
        src = FRAMES_DIR / frame_name(k)
        if src.exists():
            dst = CLEAN_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    rows = []

    # 6) Run pairwise analysis over [start, end)
    for k in range(start, end):
        fnA = frame_name(k)
        fnB = frame_name(k+1)
        fpA = FRAMES_DIR / fnA
        fpB = FRAMES_DIR / fnB

        if not (fpA.exists() and fpB.exists()):
            continue

        roiA = roi_map.get(fnA, None)
        roiB = roi_map.get(fnB, None)
        if (roiA is None) or (roiB is None):
            continue

        A = cv2.imread(str(fpA)); B = cv2.imread(str(fpB))
        if A is None or B is None:
            continue
        H, W = A.shape[:2]

        # Intersect consecutive ROIs (pad)
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
        heat_path = HEAT_DIR / f"heat_{k+1:05d}.png"
        cv2.imwrite(str(heat_path), heat_vis)

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
        arrows_path = ARW_DIR / f"arrows_{k+1:05d}.png"
        cv2.imwrite(str(arrows_path), arrow_vis)

        rows.append({
            "ooi": ooi_slug,
            "k_next": k+1,
            "frameA": fnA,
            "frameB": fnB,
            "roi_box_xyxy": [rx1, ry1, rx2, ry2],
            "heat_saved": str(heat_path),
            "arrows_saved": str(arrows_path),
            "heat_max": int(heat_u8.max()),
            "any_downward": bool(np.any(down_mask))
        })

    # 7) Save a tiny viz summary + meta
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "viz_summary.csv", index=False)

    meta = {
        "ooi": ooi_slug,
        "frames_dir": str(FRAMES_DIR),
        "roi_master_csv": str(ROI_MASTER_CSV_DEFAULT),
        "roi_sliced_csv": str(ROI_OUT_CSV),
        "heat_dir": str(HEAT_DIR),
        "arrows_dir": str(ARW_DIR),
        "copied_clean_frames_dir": str(CLEAN_DIR),
        "polarity": POLARITY,
        "thr_bright_hi": DIFF_THR_BRIGHT_HI,
        "thr_bright_lo": DIFF_THR_BRIGHT_LO,
        "thr_dark_hi": DIFF_THR_DARK_HI,
        "thr_dark_lo": DIFF_THR_DARK_LO,
        "flow_mag_thr": MAG_THR,
        "flow_vy_thr": VY_THR,
        "flow_vert_ratio": VERT_RATIO,
        "range_used": [int(START_IDX), int(END_IDX)]
    }
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
