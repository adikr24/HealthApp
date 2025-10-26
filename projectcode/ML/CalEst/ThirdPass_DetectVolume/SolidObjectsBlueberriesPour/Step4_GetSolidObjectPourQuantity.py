#!/usr/bin/env python3
# Count distinct blobs after a thickness-only filter and report:
# - blob_count (per image)
# - max_thickness_px (image-wide)
# - thinnest_blob_thickness_px (min of per-blob max thicknesses)
# - thickest_blob_thickness_px (max of per-blob max thicknesses)
# Also prints a cumulative total of blobs across all images.
#
# Black/white images (thickness_vis, thickonly) go to: <out_dir>/bw_pixels/
# Red mask + color overlay stay in <out_dir>/.

from pathlib import Path
import re
from typing import List, Tuple
import sys

import numpy as np
import cv2
from PIL import Image
import pandas as pd

# ------------ EDIT DEFAULTS FOR VS CODE RUNS ------------
INPUT_DIR  = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/volume_bands/blueberry/diff_heat")
GLOB       = "diff_*.png"
T          = 15       # minimum thickness in px (kept mask = thickness >= T)
DOMINANCE  = 15       # require R - B >= DOMINANCE (low bias; no intensity gating)
RED_MIN    = 10       # tiny floor for red channel
OUTDIR     = None     # None -> <input>/thickness_counts
LABEL_BLOBS = True    # draw numbered labels on overlay for verification
# ---------------------------------------------------------

def natural_key(p: Path) -> Tuple:
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else -1, p.name)

def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def red_dominant_mask(arr: np.ndarray, dominance: int, red_min: int) -> np.ndarray:
    R = arr[..., 0].astype(np.int16)
    B = arr[..., 2].astype(np.int16)
    return (((R >= red_min) & ((R - B) >= dominance)).astype(np.uint8) * 255)

def thickness_map(mask: np.ndarray) -> np.ndarray:
    fg = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)   # float32
    return dist * 2.0

def keep_thick(thick: np.ndarray, t: float) -> np.ndarray:
    return (thick >= float(t)).astype(np.uint8) * 255

def save_overlay(arr: np.ndarray, bin_mask: np.ndarray, out_path: Path, labels=None, alpha: float = 0.35):
    color_layer = np.zeros_like(arr)
    color_layer[bin_mask > 0] = (0, 255, 255)  # cyan
    overlay = cv2.addWeighted(arr, 1.0, color_layer, alpha, 0)

    # Optional: draw numeric labels at blob centroids
    if labels:
        for lab, (cx, cy) in labels.items():
            cv2.putText(
                overlay, str(lab), (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
    Image.fromarray(overlay).save(out_path)

def process_one(fp: Path, out_dir: Path, t: int, dominance: int, red_min: int):
    arr = load_rgb(fp)
    mask_red = red_dominant_mask(arr, dominance, red_min)
    thick = thickness_map(mask_red)
    thickonly = keep_thick(thick, t)

    # Connected components on the thick-only mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thickonly, connectivity=8)
    blob_count = max(0, num_labels - 1)

    # Per-blob thickness: take max(thickness) inside each component
    per_blob_thickness = []
    label_positions = {}
    if blob_count > 0:
        for lab in range(1, num_labels):
            blob_mask = (labels == lab)
            if not np.any(blob_mask):
                continue
            per_blob_thickness.append(float(thick[blob_mask].max()))
            cx, cy = centroids[lab]
            label_positions[lab] = (cx, cy)

    # Image-wide thickness extrema
    max_thickness_px = float(thick.max()) if thick.size else 0.0
    thinnest_blob = float(min(per_blob_thickness)) if per_blob_thickness else 0.0
    thickest_blob = float(max(per_blob_thickness)) if per_blob_thickness else 0.0

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    bw_dir = out_dir / "bw_pixels"         # <--- NEW: subfolder for B/W outputs
    bw_dir.mkdir(parents=True, exist_ok=True)
    base = fp.stem

    # Keep red mask in the main folder (as requested)
    Image.fromarray(mask_red).save(out_dir / f"{base}_redmask.png")

    # Send black/white images to bw_pixels/
    t_vis = (thick / max_thickness_px * 255.0).astype(np.uint8) if max_thickness_px > 0 else np.zeros_like(mask_red)
    Image.fromarray(t_vis).save(bw_dir / f"{base}_thickness_vis.png")
    Image.fromarray(thickonly).save(bw_dir / f"{base}_thickonly_T{t}.png")

    # Keep color overlay in the main folder
    labels_for_overlay = label_positions if LABEL_BLOBS and blob_count > 0 else None
    save_overlay(arr, thickonly, out_dir / f"{base}_overlay_T{t}.png", labels=labels_for_overlay)

    # Per-blob rows (optional details)
    blob_rows = []
    if blob_count > 0:
        for lab in range(1, num_labels):
            x, y, w, h, area = stats[lab]
            cx, cy = centroids[lab]
            blob_rows.append({
                "file": fp.name,
                "label": int(lab),
                "thickness_T": int(t),
                "blob_max_thickness_px": float(thick[labels == lab].max()),
                "bbox_x": int(x), "bbox_y": int(y),
                "bbox_w": int(w), "bbox_h": int(h),
                "area_px": int(area),
                "centroid_x": float(cx), "centroid_y": float(cy),
            })

    # Summary for this file
    summary = {
        "file": fp.name,
        "thickness_T": int(t),
        "blob_count": int(blob_count),
        "max_thickness_px": round(max_thickness_px, 2),
        "thinnest_blob_thickness_px": round(thinnest_blob, 2),
        "thickest_blob_thickness_px": round(thickest_blob, 2),
        "outputs_dir": str(out_dir),
        "bw_outputs_dir": str(bw_dir),
    }
    return summary, blob_rows

def main():
    # Optional CLI overrides (still VS Code friendly)
    import argparse
    ap = argparse.ArgumentParser(description="Count blobs and report thinnest/thickest kept blob thickness.")
    ap.add_argument("input", nargs="?", default=str(INPUT_DIR), help="Folder or single image")
    ap.add_argument("--glob", default=GLOB, help="Pattern if input is a folder")
    ap.add_argument("--t", type=int, default=T, help="Min thickness (px)")
    ap.add_argument("--dominance", type=int, default=DOMINANCE, help="Require R-B >= dominance")
    ap.add_argument("--red-min", type=int, default=RED_MIN, help="Minimum red intensity")
    ap.add_argument("--outdir", default=OUTDIR, help="Output directory (default: <input>/thickness_counts)")
    args = ap.parse_args(None if len(sys.argv) > 1 else [])

    inp = Path(args.input)
    out_dir = Path(args.outdir) if args.outdir else ((inp.parent if inp.is_file() else inp) / "thickness_counts")

    if inp.is_file():
        files = [inp]
    else:
        files = sorted(inp.glob(args.glob), key=natural_key)

    if not files:
        print(f"[WARN] No files found under: {inp} (glob='{args.glob}')")
        return

    print(f"[INFO] Processing {len(files)} file(s) with T={args.t} → {out_dir}")

    summaries, per_blob_rows = [], []
    for fp in files:
        s, rows = process_one(fp, out_dir, args.t, args.dominance, args.red_min)
        summaries.append(s)
        per_blob_rows.extend(rows)
        print(f"[OK] {fp.name}: blobs={s['blob_count']}  max_th={s['max_thickness_px']}  thin_blob={s['thinnest_blob_thickness_px']}  thick_blob={s['thickest_blob_thickness_px']}")

    # Write CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_dir / "thickness_blob_counts.csv", index=False)
    if per_blob_rows:
        pd.DataFrame(per_blob_rows).to_csv(out_dir / "thickness_blob_details.csv", index=False)

    # Cumulative total
    total_blobs = sum(s["blob_count"] for s in summaries)
    print(f"\n[TOTAL] Distinct blobs across all images @ T={args.t}: {total_blobs}")

if __name__ == "__main__":
    main()
