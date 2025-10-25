#!/usr/bin/env python3
# Draw upper/lower band inside VOI, crop the band, compare ALL consecutive frames in the window.
# Now saves 4× zoomed images for clearer visualization.

from __future__ import annotations
import csv, json, re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------- Paths ----------
VOL_REF_CSV   = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/ActivityFrameSegmentation/vol_detection_reference.csv")
ROI_MASTER    = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/flow_rois.csv")
FRAMES_DIR    = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/bounding_boxes/VOIBoundingBoxes")
OUT_DIR       = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/band_debug")

# ---------- Knobs ----------
BAND_UP_FRAC  = 0.30
BAND_DN_FRAC  = 0.70
AREA_SIM_THR  = 0.98
USE_CLAHE     = True
OUTPUT_SCALE  = 4.0      # <<-- increased to 4× zoom
DRAW_COLOR_UP = (0, 255, 255)
DRAW_COLOR_DN = (255, 255, 0)
DRAW_THICK    = 2

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

def gamma_LUT(gamma=0.8):
    x = np.arange(256, dtype=np.float32)/255.0
    y = np.power(x, gamma)
    return (np.clip(y*255,0,255)).astype(np.uint8)
_GAMMA = gamma_LUT(0.8)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) if USE_CLAHE else None

def to_gray(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    if _CLAHE is not None: L = _CLAHE.apply(L)
    return cv2.LUT(L, _GAMMA)

def build_blue2red_LUT():
    lut = np.zeros((256,1,3), dtype=np.uint8)
    for i in range(256):
        t = i/255.0
        b = int(round(255*(1.0 - t)))
        r = int(round(255*t))
        lut[i,0,:] = (b,0,r)
    return lut
BLUE2RED_LUT = build_blue2red_LUT()

def colorize_heat(gray_0to255):
    g = np.clip(gray_0to255, 0, 255).astype(np.uint8)
    return cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), BLUE2RED_LUT)

# ---------- ROI logic ----------
def choose_window(vol_csv: Path) -> Tuple[int,int]:
    df = pd.read_csv(vol_csv)
    if "start_idx" not in df.columns or "end_idx" not in df.columns:
        raise ValueError("vol_detection_reference.csv missing start_idx/end_idx")
    row = None
    if "OOI" in df.columns:
        mask = df["OOI"].astype(str).str.lower()=="blueberry"
        if mask.any(): row = df.loc[mask].iloc[0]
    if row is None and (df["classes"].astype(str).str.contains("blueberry", case=False, na=False)).any():
        row = df[df["classes"].astype(str).str.contains("blueberry", case=False, na=False)].iloc[0]
    if row is None: row = df.iloc[0]
    return int(row["start_idx"]), int(row["end_idx"])

def slice_roi(master_csv: Path, out_csv: Path, start_idx: int, end_idx: int):
    df = pd.read_csv(master_csv)
    need = {"filename","roi_x1","roi_y1","roi_x2","roi_y2"}
    if not need.issubset(df.columns):
        raise ValueError("ROI CSV must have filename, roi_x1, roi_y1, roi_x2, roi_y2")
    def idx_from_row(fn:str):
        try: return int(Path(fn).stem.split("_")[1])
        except: return -1
    df["__idx"] = df["filename"].astype(str).map(idx_from_row)
    sliced = df[(df["__idx"]>=start_idx) & (df["__idx"]<=end_idx)].drop(columns="__idx")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out_csv, index=False)
    return out_csv

def load_roi(path: Path):
    roi = {}
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fn = r.get("filename","")
            if not fn: continue
            x1=r.get("roi_x1",""); y1=r.get("roi_y1","")
            x2=r.get("roi_x2",""); y2=r.get("roi_y2","")
            if "" in (x1,y1,x2,y2): continue
            roi[fn] = (int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))
    return roi

def build_sequence_from_roi_and_dir(roi_csv_path: Path, frames_dir: Path) -> List[str]:
    df = pd.read_csv(roi_csv_path)
    uniq, seen = [], set()
    for fn in df["filename"].astype(str).tolist():
        if fn in seen: continue
        p = frames_dir / fn
        if p.exists():
            uniq.append(fn); seen.add(fn)
        else:
            cand = list(frames_dir.glob(Path(fn).stem + ".*"))
            if cand:
                uniq.append(cand[0].name); seen.add(cand[0].name)
    uniq.sort(key=idx_from_fn)
    return uniq

# ---------- Band math ----------
def compute_band_dual(y1:int, y2:int):
    h = max(1, y2 - y1)
    y_up_loc = int(BAND_UP_FRAC * h)
    y_dn_loc = int(BAND_DN_FRAC * h)
    y_up_abs = y1 + y_up_loc
    y_dn_abs = y1 + y_dn_loc
    return y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, h

# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAW_DIR   = OUT_DIR / "bands_drawn"; DRAW_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR   = OUT_DIR / "band_crops";  CROP_DIR.mkdir(parents=True, exist_ok=True)
    DIFF_DIR   = OUT_DIR / "diff_heat";   DIFF_DIR.mkdir(parents=True, exist_ok=True)

    s, e = choose_window(VOL_REF_CSV)
    roi_csv = OUT_DIR / "flow_rois.csv"
    slice_roi(ROI_MASTER, roi_csv, s, e)
    roi_map = load_roi(roi_csv)
    seq = build_sequence_from_roi_and_dir(roi_csv, FRAMES_DIR)
    if len(seq) < 2:
        print(json.dumps({"error":"need at least 2 frames in window","seq_len":len(seq)}, indent=2))
        return

    csv_rows = []

    def draw_bands(full_img, x1,y1,x2,y2, tag):
        y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, h = compute_band_dual(y1,y2)
        viz = full_img.copy()
        cv2.rectangle(viz, (x1,y1), (x2,y2), (255,128,0), 2)
        cv2.line(viz, (x1, y_up_abs), (x2, y_up_abs), DRAW_COLOR_UP, DRAW_THICK, cv2.LINE_AA)
        cv2.line(viz, (x1, y_dn_abs), (x2, y_dn_abs), DRAW_COLOR_DN, DRAW_THICK, cv2.LINE_AA)
        # upscale 4× before saving
        viz_big = cv2.resize(viz, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
        out = DRAW_DIR / f"bands_{tag}.jpg"
        cv2.imwrite(str(out), viz_big)
        return y_up_loc, y_dn_loc, out

    def save_scaled(img, path):
        vis = cv2.resize(img, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), vis)

    for i in range(len(seq) - 1):
        fnA, fnB = seq[i], seq[i+1]
        pA, pB   = FRAMES_DIR / fnA, FRAMES_DIR / fnB
        A, B     = imread_any(pA), imread_any(pB)
        if A is None or B is None:
            csv_rows.append({"frameA": fnA, "frameB": fnB, "error": "failed to read frame(s)"})
            continue

        roiA = roi_map.get(fnA) or roi_map.get(fnB)
        roiB = roi_map.get(fnB) or roi_map.get(fnA)
        if roiA is None or roiB is None:
            csv_rows.append({"frameA": fnA, "frameB": fnB, "error": "missing ROI"})
            continue

        x1a,y1a,x2a,y2a = roiA
        x1b,y1b,x2b,y2b = roiB
        tagA = f"{idx_from_fn(fnA):05d}"
        tagB = f"{idx_from_fn(fnB):05d}"
        y_upA_loc, y_dnA_loc, drawA = draw_bands(A, x1a,y1a,x2a,y2a, tagA)
        y_upB_loc, y_dnB_loc, drawB = draw_bands(B, x1b,y1b,x2b,y2b, tagB)

        A_roi, B_roi = A[y1a:y2a, x1a:x2a], B[y1b:y2b, x1b:x2b]
        A_band, B_band = A_roi[y_upA_loc:y_dnA_loc, :], B_roi[y_upB_loc:y_dnB_loc, :]
        if A_band.size == 0 or B_band.size == 0:
            csv_rows.append({"frameA": fnA, "frameB": fnB, "error": "empty band"})
            continue

        areaA = A_band.shape[0]*A_band.shape[1]
        areaB = B_band.shape[0]*B_band.shape[1]
        area_sim = min(areaA, areaB)/max(areaA, areaB)

        cropA_path = CROP_DIR / f"crop_{tagA}.png"
        cropB_path = CROP_DIR / f"crop_{tagB}.png"
        save_scaled(A_band, cropA_path)
        save_scaled(B_band, cropB_path)

        diff_path = ""
        if area_sim >= AREA_SIM_THR:
            if B_band.shape[:2] != A_band.shape[:2]:
                B_band = cv2.resize(B_band, (A_band.shape[1], A_band.shape[0]), interpolation=cv2.INTER_LINEAR)
            gA, gB = to_gray(A_band), to_gray(B_band)
            diff = cv2.absdiff(gB, gA)
            diff_norm = (diff.astype(np.float32)/float(diff.max())*255.0).astype(np.uint8) if diff.max()>0 else diff.copy()
            heat = colorize_heat(diff_norm)
            heat_big = cv2.resize(heat, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
            diff_path = str(DIFF_DIR / f"diff_{tagA}_{tagB}.png")
            cv2.imwrite(diff_path, heat_big)

        csv_rows.append({
            "frameA": fnA, "frameB": fnB,
            "roiA_xyxy": [x1a,y1a,x2a,y2a],
            "roiB_xyxy": [x1b,y1b,x2b,y2b],
            "area_similarity": round(area_sim,4),
            "band_crop_A": str(cropA_path),
            "band_crop_B": str(cropB_path),
            "bands_drawn_A": str(drawA),
            "bands_drawn_B": str(drawB),
            "diff_heatmap": diff_path
        })

    out_csv = OUT_DIR / "band_compare_summary.csv"
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(json.dumps({
        "done": True,
        "num_pairs": len(csv_rows),
        "first_frame": seq[0],
        "last_frame": seq[-1],
        "zoom": OUTPUT_SCALE,
        "csv": str(out_csv)
    }, indent=2))

if __name__ == "__main__":
    main()
