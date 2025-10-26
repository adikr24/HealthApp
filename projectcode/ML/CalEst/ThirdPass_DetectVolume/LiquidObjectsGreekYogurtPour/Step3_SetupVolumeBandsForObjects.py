#!/usr/bin/env python3
# Minimal band visualizer:
# - slice flow_rois.csv for window
# - draw ROI + upper/lower band lines (bands_drawn/)
# - save band crops (band_crops/)
# No white masks, no heatmaps, no metrics.

from __future__ import annotations
import csv, json, re, bisect
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------- Paths (greekyogurt) ----------
VOL_REF_CSV   = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/ActivityFrameSegmentation/vol_detection_reference.csv")
ROI_MASTER    = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/VOIYolo_ROI/flow_rois.csv")
FRAMES_DIR    = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/VOIYolo_ROI/")
OUT_DIR       = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/greek_yogurt/")

# ---------- Knobs ----------
OOI_NAME      = "greek_yogurt"  # object-of-interest to match in vol_detection_reference (optional)
STATE_INDEX   = 1               # 0-based; 1 = "second state"

# Band geometry (fractions of ROI height)
BAND_UP_FRAC  = 0.30
BAND_DN_FRAC  = 0.70

# Viz
OUTPUT_SCALE  = 4.0
DRAW_COLOR_UP = (0, 255, 255)   # cyan/yellow-ish
DRAW_COLOR_DN = (255, 255, 0)
ROI_COLOR     = (255, 128, 0)
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

def compute_band_dual(y1:int, y2:int):
    h = max(1, y2 - y1)
    y_up_loc = int(BAND_UP_FRAC * h)
    y_dn_loc = int(BAND_DN_FRAC * h)
    y_up_abs = y1 + y_up_loc
    y_dn_abs = y1 + y_dn_loc
    return y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, h

def choose_window_second_state(vol_csv: Path, ooi_name: Optional[str], state_index: int) -> Tuple[int,int]:
    """
    Priority:
      1) rows with OOI == ooi_name, pick row at state_index
      2) else rows where 'classes' contains ooi_name, pick state_index
      3) else pick state_index overall
      4) fallback to first row
    """
    df = pd.read_csv(vol_csv)
    if "start_idx" not in df.columns or "end_idx" not in df.columns:
        raise ValueError("vol_detection_reference.csv missing start_idx/end_idx")

    def pick_row(sub: pd.DataFrame) -> Optional[pd.Series]:
        if sub.empty: return None
        i = max(0, min(state_index, len(sub)-1))
        return sub.iloc[i]

    row = None
    if ooi_name and "OOI" in df.columns:
        row = pick_row(df.loc[df["OOI"].astype(str).str.lower() == str(ooi_name).lower()])
    if row is None and ooi_name and "classes" in df.columns:
        row = pick_row(df.loc[df["classes"].astype(str).str.contains(str(ooi_name), case=False, na=False)])
    if row is None:
        row = pick_row(df)
    if row is None:
        row = df.iloc[0]

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

def load_roi(path: Path) -> Dict[str, Tuple[int,int,int,int]]:
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

# ---- Drive frames by files in [s,e] + ROI fallback (carry-forward + nearest) ----
def list_frames_in_dir_bounded(frames_dir: Path, start_idx: int, end_idx: int) -> List[str]:
    cands = list(frames_dir.glob("frame_*.jpg")) + list(frames_dir.glob("frame_*.png"))
    out = []
    for p in cands:
        m = re.search(r"(\d+)", p.stem)
        idx = int(m.group(1)) if m else -1
        if start_idx <= idx <= end_idx:
            out.append((idx, p.name))
    out.sort(key=lambda t: t[0])
    return [name for _, name in out]

def build_roi_index(roi_map: dict) -> dict:
    idx_map = {}
    for fn, xyxy in roi_map.items():
        m = re.search(r"(\d+)", Path(fn).stem)
        if m:
            idx_map[int(m.group(1))] = xyxy
    return idx_map

def get_roi_with_fallback(frame_idx: int, roi_index: dict, last_roi: Optional[tuple]) -> Optional[tuple]:
    if frame_idx in roi_index:
        return roi_index[frame_idx]
    if last_roi is not None:
        return last_roi
    if roi_index:
        keys = sorted(roi_index.keys())
        pos = bisect.bisect_left(keys, frame_idx)
        candidates = []
        if pos < len(keys):   candidates.append(keys[pos])
        if pos > 0:           candidates.append(keys[pos-1])
        if candidates:
            best_k = min(candidates, key=lambda k: abs(k - frame_idx))
            return roi_index[best_k]
    return None

def save_scaled(img, path: Path, scale: float):
    vis = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), vis)

# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAW_DIR   = OUT_DIR / "bands_drawn"; DRAW_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR   = OUT_DIR / "band_crops";  CROP_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Choose window (second state for greek_yogurt)
    s, e = choose_window_second_state(VOL_REF_CSV, OOI_NAME, STATE_INDEX)

    # 2) Create sliced flow_rois.csv (for auditing)
    roi_csv = OUT_DIR / "flow_rois.csv"
    slice_roi(ROI_MASTER, roi_csv, s, e)
    roi_map = load_roi(roi_csv)

    # 3) Build frame list by files present in directory in [s,e]
    seq = list_frames_in_dir_bounded(FRAMES_DIR, s, e)
    if len(seq) == 0:
        print(json.dumps({"error":"no frames in window (by files)", "start_idx": s, "end_idx": e}, indent=2))
        return

    # 4) ROI index + carry-forward fallback
    roi_index = build_roi_index(roi_map)
    last_roi: Optional[tuple] = None

    drawn, cropped, skipped = 0, 0, 0

    # 5) Per-frame: draw ROI + band lines, save band crop
    for fn in seq:
        fidx = idx_from_fn(fn)
        p = FRAMES_DIR / fn
        img = imread_any(p)
        if img is None:
            skipped += 1
            continue

        roi = get_roi_with_fallback(fidx, roi_index, last_roi)
        if roi is None:
            skipped += 1
            continue
        last_roi = roi

        x1,y1,x2,y2 = roi
        y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, _h = compute_band_dual(y1,y2)

        # Guard tiny/degenerate bands: force minimal height
        if y_dn_loc <= y_up_loc:
            y_dn_loc = min(max(1, y2 - y1), y_up_loc + 2)
            y_dn_abs = y1 + y_dn_loc

        # Draw on full image
        viz = img.copy()
        cv2.rectangle(viz, (x1,y1), (x2,y2), ROI_COLOR, 2)
        cv2.line(viz, (x1, y_up_abs), (x2, y_up_abs), DRAW_COLOR_UP, DRAW_THICK, cv2.LINE_AA)
        cv2.line(viz, (x1, y_dn_abs), (x2, y_dn_abs), DRAW_COLOR_DN, DRAW_THICK, cv2.LINE_AA)

        draw_path = DRAW_DIR / f"bands_{fidx:05d}.jpg"
        save_scaled(viz, draw_path, OUTPUT_SCALE)
        drawn += 1

        # Band crop (rows y_up..y_dn within ROI)
        roi_img = img[y1:y2, x1:x2]
        band = roi_img[y_up_loc:y_dn_loc, :]
        if band.size == 0:
            skipped += 1
            continue

        crop_path = CROP_DIR / f"crop_{fidx:05d}.png"
        save_scaled(band, crop_path, OUTPUT_SCALE)
        cropped += 1

    # 6) Summary
    print(json.dumps({
        "done": True,
        "ooi": OOI_NAME,
        "state_index": STATE_INDEX,
        "start_idx": s,
        "end_idx": e,
        "frames_seen": len(seq),
        "bands_drawn": drawn,
        "band_crops": cropped,
        "skipped": skipped,
        "flow_rois_csv": str(roi_csv),
        "out_dirs": {
            "bands_drawn": str(DRAW_DIR),
            "band_crops": str(CROP_DIR),
        }
    }, indent=2))

if __name__ == "__main__":
    main()
