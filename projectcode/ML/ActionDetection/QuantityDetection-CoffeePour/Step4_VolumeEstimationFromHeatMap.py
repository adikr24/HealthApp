#!/usr/bin/env python3
# Frame-by-frame band analysis with continuous differencing (EMA + hysteresis).
# Enumerates frames from directory within [start_idx, end_idx] and uses ROI fallback.
# Tailored for "greek_yogurt": selects the *second state* from vol_detection_reference.csv.

from __future__ import annotations
import csv, json, re, bisect
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
OOI_NAME      = "greek_yogurt"  # object-of-interest name to match (optional; falls back safely)
STATE_INDEX   = 1               # <-- "second state" (0: first, 1: second)

# Band geometry (as fractions of ROI height)
BAND_UP_FRAC  = 0.30
BAND_DN_FRAC  = 0.70

# Image processing
USE_CLAHE     = True
OUTPUT_SCALE  = 4.0
DRAW_COLOR_UP = (0, 255, 255)
DRAW_COLOR_DN = (255, 255, 0)
DRAW_THICK    = 2

# White-pixel focus (milk/yogurt)
WHITE_THR     = 230   # luminance (0..255) threshold to consider "white"

# Continuous differencing (EMA + hysteresis)
EMA_ALPHA       = 0.30     # smoothing for p_t (0..1). Higher = more reactive
DELTA_START_THR = 0.004    # start if Δ_t > this for START_CONSEC frames (normalized units)
DELTA_STOP_THR  = 0.002    # consider "idle" if |Δ_t| <= this for STOP_CONSEC frames
START_CONSEC    = 3        # frames of Δ_t > start threshold to "enter pouring"
STOP_CONSEC     = 6        # frames of |Δ_t| <= stop threshold to "exit pouring"

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
    if _CLAHE is not None:
        L = _CLAHE.apply(L)
    return cv2.LUT(L, _GAMMA)

def compute_band_dual(y1:int, y2:int):
    h = max(1, y2 - y1)
    y_up_loc = int(BAND_UP_FRAC * h)
    y_dn_loc = int(BAND_DN_FRAC * h)
    y_up_abs = y1 + y_up_loc
    y_dn_abs = y1 + y_dn_loc
    return y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, h

def white_mask_from_gray(gray: np.ndarray, thr: int) -> np.ndarray:
    return (gray >= thr).astype(np.uint8) * 255

def count_white(mask_0_255: np.ndarray) -> int:
    return int((mask_0_255 > 0).sum())

def line_white_count(mask_0_255: np.ndarray, row_idx: int) -> int:
    if row_idx < 0 or row_idx >= mask_0_255.shape[0]: return 0
    return int((mask_0_255[row_idx, :] > 0).sum())

def choose_window_second_state(vol_csv: Path, ooi_name: Optional[str], state_index: int) -> Tuple[int,int]:
    """
    Window selection priority:
      1) rows with OOI == ooi_name (case-insensitive), pick row at state_index
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
        mask = df["OOI"].astype(str).str.lower() == str(ooi_name).lower()
        row = pick_row(df.loc[mask])
    if row is None and ooi_name and "classes" in df.columns:
        mask = df["classes"].astype(str).str.contains(str(ooi_name), case=False, na=False)
        row = pick_row(df.loc[mask])
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

# -------- NEW: drive frames by files and ROI fallback --------
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

# ---------- Main ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DRAW_DIR   = OUT_DIR / "bands_drawn"; DRAW_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR   = OUT_DIR / "band_crops";  CROP_DIR.mkdir(parents=True, exist_ok=True)
    WHITE_DIR  = OUT_DIR / "white_masks"; WHITE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Choose window (second state for greek_yogurt)
    s, e = choose_window_second_state(VOL_REF_CSV, OOI_NAME, STATE_INDEX)

    # Keep a sliced ROI CSV for auditing, but do NOT drive the frame list from it
    roi_csv = OUT_DIR / "flow_rois.csv"
    slice_roi(ROI_MASTER, roi_csv, s, e)
    roi_map = load_roi(roi_csv)

    # Drive by files present in directory
    seq = list_frames_in_dir_bounded(FRAMES_DIR, s, e)
    if len(seq) == 0:
        print(json.dumps({"error":"no frames in window (by files)", "start_idx": s, "end_idx": e}, indent=2)); return

    # Build ROI index and init carry-forward
    roi_index = build_roi_index(roi_map)
    last_roi: Optional[tuple] = None

    per_frame_rows: List[Dict[str, Any]] = []

    def save_scaled(img, path: Path):
        vis = cv2.resize(img, None, fx=OUTPUT_SCALE, fy=OUTPUT_SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(path), vis)

    # 2) Per-frame pass
    for fn in seq:
        fidx = idx_from_fn(fn)
        p = FRAMES_DIR / fn
        img = imread_any(p)
        if img is None:
            per_frame_rows.append({"frame": fn, "frame_idx": fidx, "error": "failed to read frame"})
            continue

        roi = get_roi_with_fallback(fidx, roi_index, last_roi)
        if roi is None:
            per_frame_rows.append({"frame": fn, "frame_idx": fidx, "error": "missing ROI (after fallback)"})
            continue
        last_roi = roi  # carry-forward

        x1,y1,x2,y2 = roi
        y_up_loc, y_dn_loc, y_up_abs, y_dn_abs, _h = compute_band_dual(y1,y2)

        # Guard tiny/degenerate bands: force at least 2 rows if needed
        if y_dn_loc <= y_up_loc:
            y_dn_loc = min(max(1, y2 - y1), y_up_loc + 2)

        # Draw bands on full image
        viz = img.copy()
        cv2.rectangle(viz, (x1,y1), (x2,y2), (255,128,0), 2)
        cv2.line(viz, (x1, y_up_abs), (x2, y_up_abs), DRAW_COLOR_UP, DRAW_THICK, cv2.LINE_AA)
        cv2.line(viz, (x1, y_dn_abs), (x2, y_dn_abs), DRAW_COLOR_DN, DRAW_THICK, cv2.LINE_AA)
        draw_path = DRAW_DIR / f"bands_{fidx:05d}.jpg"
        save_scaled(viz, draw_path)

        # Band crop (within ROI, rows y_up..y_dn)
        roi_img = img[y1:y2, x1:x2]
        band = roi_img[y_up_loc:y_dn_loc, :]
        if band.size == 0:
            per_frame_rows.append({"frame": fn, "frame_idx": fidx, "error": "empty band", "roi_xyxy":[x1,y1,x2,y2]})
            continue

        crop_path = CROP_DIR / f"crop_{fidx:05d}.png"
        save_scaled(band, crop_path)

        # Luminance + white mask
        g = to_gray(band)
        white = white_mask_from_gray(g, WHITE_THR)
        white_path = WHITE_DIR / f"white_{fidx:05d}.png"
        save_scaled(white, white_path)

        H_band, W_band = white.shape[:2]
        band_area = float(H_band * W_band)
        white_count = count_white(white)
        white_ratio = (white_count / band_area) if band_area > 0 else 0.0

        # Tripwire lines inside band coords:
        y_up_row_band = 0
        y_dn_row_band = max(0, H_band - 1)
        line_up = line_white_count(white, y_up_row_band)
        line_dn = line_white_count(white, y_dn_row_band)

        per_frame_rows.append({
            "frame": fn,
            "frame_idx": fidx,
            "roi_xyxy": [x1,y1,x2,y2],
            "band_rows_local": [int(y_up_loc), int(y_dn_loc)],
            "band_crop": str(crop_path),
            "bands_drawn": str(draw_path),
            "white_mask": str(white_path),
            "white_thr": WHITE_THR,
            "band_h": int(H_band),
            "band_w": int(W_band),
            "band_area": int(band_area),
            "white_count": int(white_count),
            "white_ratio": float(white_ratio),
            "line_up_white": int(line_up),
            "line_dn_white": int(line_dn),
        })

    # 3) Build DataFrame and compute EMA + deltas over valid rows
    df = pd.DataFrame(per_frame_rows)
    out_per_frame = OUT_DIR / "band_per_frame_metrics_greekyogurt.csv"
    if df.empty:
        df.to_csv(out_per_frame, index=False)
        print(json.dumps({"done": True, "ooi": OOI_NAME, "state_index": STATE_INDEX,
                          "white_thr": WHITE_THR, "frames": 0, "csv": str(out_per_frame)}, indent=2))
        return

    ok = ~(df.get("error").notna()) if "error" in df.columns else pd.Series([True]*len(df))
    df_ok = df[ok].copy().sort_values("frame_idx")

    ema_vals = []
    last = None
    for v in df_ok["white_ratio"].astype(float).tolist():
        last = v if last is None else (EMA_ALPHA * v + (1.0 - EMA_ALPHA) * last)
        ema_vals.append(last)
    df_ok["white_ratio_ema"] = ema_vals

    deltas = [0.0]
    for i in range(1, len(df_ok)):
        deltas.append(df_ok["white_ratio_ema"].iloc[i] - df_ok["white_ratio_ema"].iloc[i-1])
    df_ok["delta"] = deltas

    df = df.merge(df_ok[["frame","white_ratio_ema","delta"]], on="frame", how="left")

    # 4) Hysteresis: contiguous "pour" segments
    segments: List[Dict[str, Any]] = []
    in_seg = False
    start_i = None
    pos_run = 0
    idle_run = 0

    frames_ok = df_ok["frame"].tolist()
    idx_ok    = df_ok["frame_idx"].tolist()
    ema_ok    = df_ok["white_ratio_ema"].tolist()
    delta_ok  = df_ok["delta"].tolist()

    for i in range(len(df_ok)):
        d = float(delta_ok[i])
        pos_run = pos_run + 1 if d > DELTA_START_THR else 0
        idle_run = idle_run + 1 if abs(d) <= DELTA_STOP_THR else 0

        if not in_seg and pos_run >= START_CONSEC:
            in_seg = True
            start_i = max(0, i - START_CONSEC + 1)
            idle_run = 0

        if in_seg and idle_run >= STOP_CONSEC:
            end_i = max(start_i, i - STOP_CONSEC)
            seg = {
                "start_frame": frames_ok[start_i],
                "end_frame": frames_ok[end_i],
                "start_idx": int(idx_ok[start_i]),
                "end_idx": int(idx_ok[end_i]),
                "frames": int(end_i - start_i + 1),
                "white_ratio_ema_start": float(ema_ok[start_i]),
                "white_ratio_ema_end": float(ema_ok[end_i]),
                "delta_white_ratio": float(ema_ok[end_i] - ema_ok[start_i]),
                "mean_delta": float(np.mean(delta_ok[start_i+1:end_i+1]) if end_i > start_i else 0.0),
            }
            segments.append(seg)
            in_seg = False
            start_i = None
            pos_run = 0
            idle_run = 0

    if in_seg and start_i is not None:
        end_i = len(df_ok) - 1
        seg = {
            "start_frame": frames_ok[start_i],
            "end_frame": frames_ok[end_i],
            "start_idx": int(idx_ok[start_i]),
            "end_idx": int(idx_ok[end_i]),
            "frames": int(end_i - start_i + 1),
            "white_ratio_ema_start": float(ema_ok[start_i]),
            "white_ratio_ema_end": float(ema_ok[end_i]),
            "delta_white_ratio": float(ema_ok[end_i] - ema_ok[start_i]),
            "mean_delta": float(np.mean(delta_ok[start_i+1:end_i+1]) if end_i > start_i else 0.0),
        }
        segments.append(seg)

    # 5) Save outputs
    df.to_csv(out_per_frame, index=False)
    out_segments = OUT_DIR / "pour_segments_greekyogurt.csv"
    pd.DataFrame(segments).to_csv(out_segments, index=False)

    print(json.dumps({
        "done": True,
        "mode": "per-frame continuous",
        "ooi": OOI_NAME,
        "state_index": STATE_INDEX,
        "white_thr": WHITE_THR,
        "ema_alpha": EMA_ALPHA,
        "delta_start_thr": DELTA_START_THR,
        "delta_stop_thr": DELTA_STOP_THR,
        "start_consec": START_CONSEC,
        "stop_consec": STOP_CONSEC,
        "frames_total": len(df),
        "frames_valid": int(df_ok.shape[0]),
        "segments_found": len(segments),
        "per_frame_csv": str(out_per_frame),
        "segments_csv": str(out_segments),
        "first_frame": seq[0],
        "last_frame": seq[-1]
    }, indent=2))

if __name__ == "__main__":
    main()
