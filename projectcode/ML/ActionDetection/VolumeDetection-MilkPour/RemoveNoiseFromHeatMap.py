## Step 3 of 3: Remove rim arcs from heat map frames, keeping all flow pixels

# filter_rims_keep_all_flow.py
import os, csv, glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# ====== INPUTS ======
HEAT_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/milk_vol_heat_and_arrows/roi_pixel_metrics/heat_frames"
OUT_DIR  = os.path.join(str(Path(HEAT_DIR).parent), "rim_filtered_keep_all_flow")
os.makedirs(OUT_DIR, exist_ok=True)

START_IDX, END_IDX = None, None           # set to ints if you want a subset
FNAME_FMT = "heat_{:05d}.png"

# ====== Red extraction & blob tracing ======
RED_THR = 6            # VERY LOW: we keep all legitimate red; just need > blue background
MIN_PIXELS = 12        # ignore tiny specks only
TOP_BAND_FRAC = 0.15   # top band (rim zone) height as fraction of image height

# ====== Rim (arc) vs Flow (stream) decision thresholds ======
# Angle from vertical (0° = vertical stream, 90° = horizontal rim)
ANGLE_RIM_MIN = 55.0        # ≥ this ⇒ likely horizontal/arc
# “Tortuosity”: perimeter vs diameter (higher = curvy arc)
TORTUOSITY_RIM = 1.60
# Shape span test: flat band near top
Y_SPAN_MAX = 0.25           # small vertical span
X_SPAN_MIN = 0.35           # wide horizontal span
# Position dominance near rim
TOP_DOM_FRAC = 0.50         # ≥ this of pixels inside top band ⇒ likely rim

CSV_DECISIONS = os.path.join(OUT_DIR, "rim_decisions.csv")

def list_indices():
    paths = sorted(glob.glob(os.path.join(HEAT_DIR, "heat_*.png")))
    ids = []
    for p in paths:
        try: ids.append(int(Path(p).stem.split("_")[1]))
        except: pass
    if not ids: return []
    lo = min(ids) if START_IDX is None else START_IDX
    hi = max(ids) if END_IDX   is None else END_IDX
    return list(range(lo, hi+1))

def pca_angle(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size < 2: return 90.0
    X = np.column_stack([xs, ys]).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    C = np.cov(X.T)
    w, V = np.linalg.eig(C)
    vmaj = V[:, np.argmax(w)]
    vertical = np.array([0.0, 1.0], dtype=np.float32)
    cosang = abs(np.dot(vmaj/np.linalg.norm(vmaj), vertical))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def tortuosity(mask_bool):
    m = (mask_bool.astype(np.uint8)*255)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return 1.0
    c = max(cnts, key=cv2.contourArea)
    per = float(cv2.arcLength(c, True))
    pts = c.reshape(-1,2)
    # diameter ~ max pairwise distance via extreme points (approx by hull)
    hull = cv2.convexHull(pts)
    hull = hull.reshape(-1,2)
    diam = 1.0
    for i in range(len(hull)):
        for j in range(i+1, len(hull)):
            d = np.linalg.norm(hull[i]-hull[j])
            if d > diam: diam = d
    return per / (diam + 1e-6)

def classify_blob(mask_bool, H, W):
    area = int(mask_bool.sum())
    if area < MIN_PIXELS:
        return "ignore", {"area": area}

    ys, xs = np.where(mask_bool)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y_span = (y1 - y0 + 1) / max(1, H)
    x_span = (x1 - x0 + 1) / max(1, W)

    # position feature: dominance in the top band
    rim_y = int(round(TOP_BAND_FRAC * H))
    top_frac = float((ys < rim_y).sum()) / max(1, area)

    # orientation & shape
    ang = pca_angle(mask_bool)             # deg from vertical
    tort = tortuosity(mask_bool)           # perimeter/diameter

    is_rim = (
        (ang >= ANGLE_RIM_MIN) or
        (top_frac >= TOP_DOM_FRAC and tort >= TORTUOSITY_RIM) or
        (y_span <= Y_SPAN_MAX and x_span >= X_SPAN_MIN)
    )

    label = "rim" if is_rim else "flow"
    details = {
        "area": area, "top_frac": round(top_frac,3),
        "angle_deg": round(ang,1), "tortuosity": round(tort,3),
        "y_span": round(y_span,3), "x_span": round(x_span,3),
        "bbox": [int(x0), int(y0), int(x1), int(y1)]
    }
    return label, details

def main():
    rows = []
    idxs = list_indices()
    if not idxs:
        print("No heat frames found.")
        return

    for idx in idxs:
        fp = os.path.join(HEAT_DIR, FNAME_FMT.format(idx))
        heat = cv2.imread(fp, cv2.IMREAD_COLOR)
        if heat is None:
            continue

        H, W = heat.shape[:2]
        red = heat[...,2].astype(np.uint8)

        # Very permissive red mask: keep essentially all red pixels
        mask = (red > RED_THR).astype(np.uint8)

        # Trace blobs (8-connected)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            # Nothing to filter; copy as-is
            cv2.imwrite(os.path.join(OUT_DIR, Path(fp).name), heat)
            continue

        keep_mask = np.zeros((H, W), np.uint8)
        rim_mask  = np.zeros((H, W), np.uint8)

        for lab in range(1, num):
            comp = (labels == lab)
            label, det = classify_blob(comp, H, W)

            det.update({"frame_idx": idx, "blob_id": lab, "label": label})
            rows.append(det)

            if label == "rim":
                rim_mask[comp] = 255
            elif label == "flow":
                keep_mask[comp] = 255
            # "ignore" (too small) → drop silently

        # Build output image: keep original reds for flow; remove rim pixels
        out = heat.copy()
        # Set rim pixels to background blue (remove entirely, no dim on stream)
        # Estimate background as median of blue channel on non-red pixels:
        bg_b = int(np.median(heat[...,0][red <= RED_THR])) if np.any(red <= RED_THR) else 64
        out[rim_mask>0] = (bg_b, 0, 0)

        cv2.imwrite(os.path.join(OUT_DIR, Path(fp).name), out)

    if rows:
        df = pd.DataFrame(rows, columns=[
            "frame_idx","blob_id","label","area","top_frac","angle_deg",
            "tortuosity","y_span","x_span","bbox"
        ])
        df.to_csv(CSV_DECISIONS, index=False)
        print(f"Saved rim/flow decisions → {CSV_DECISIONS}")
        print(f"Filtered frames (rims removed, flows untouched) → {OUT_DIR}")
    else:
        print("No blobs found / nothing to log.")

if __name__ == "__main__":
    main()
