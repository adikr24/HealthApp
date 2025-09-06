# filter_rims_keep_all_flow.py
# Step 3 of 3: Remove rim arcs from heat map frames, keeping stream-like pixels

import os, glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# ====== INPUTS ======
HEAT_DIR = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/coffee_vol_heat_and_arrows/roi_pixel_metrics/heat_frames"
OUT_DIR  = "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/coffee_vol_heat_and_arrows/roi_pixel_metrics/rim_filtered_keep_all_flow"
os.makedirs(OUT_DIR, exist_ok=True)

START_IDX, END_IDX = None, None           # set to ints for a subset
FNAME_FMT = "heat_{:05d}.png"

# ====== Red extraction & blob tracing ======
RED_THR      = 6      # keep essentially all red pixels (> deep blue background)
MIN_PIXELS   = 12     # ignore tiny specks

# Top band where rim glare lives (relative to image height)
TOP_BAND_FRAC = 0.15

# ====== FLOW acceptance (stream-like) ======
# Angle from vertical (0° = perfectly vertical, 90° = horizontal)
ANGLE_FLOW_MAX    = 30.0   # <= this → likely a falling stream (allow a bit of slant)
Y_SPAN_MIN_FLOW   = 0.30   # blob spans at least 30% of image height
TOP_FRAC_MAX_FLOW = 0.50   # no more than half of the blob lives in the top band

# Optional: require that the blob touches a thin gate band just below the rim
REQUIRE_GATE_TOUCH = True
GATE_OFFSET_FRAC   = 0.08  # distance below image top (as frac of H)
GATE_BAND_FRAC     = 0.06  # band thickness (as frac of H)

# ====== RIM rejection (arc/splash-like) ======
ANGLE_RIM_MIN  = 55.0   # >= this → horizontal/arc
TORTUOSITY_RIM = 1.60   # perimeter / diameter (curvier arcs are larger)
Y_SPAN_MAX     = 0.25   # shallow vertical span
X_SPAN_MIN     = 0.35   # wide horizontal span
TOP_DOM_FRAC   = 0.50   # majority of pixels in top band

CSV_DECISIONS = os.path.join(OUT_DIR, "rim_decisions.csv")

# ------------------ helpers ------------------

def list_indices():
    paths = sorted(glob.glob(os.path.join(HEAT_DIR, "heat_*.png")))
    ids = []
    for p in paths:
        try:
            ids.append(int(Path(p).stem.split("_")[1]))
        except Exception:
            pass
    if not ids:
        return []
    lo = min(ids) if START_IDX is None else START_IDX
    hi = max(ids) if END_IDX   is None else END_IDX
    return list(range(lo, hi + 1))

def pca_angle(mask_bool):
    """Return angle (deg) from vertical of the blob's major axis."""
    ys, xs = np.where(mask_bool)
    if xs.size < 2:
        return 90.0
    X = np.column_stack([xs, ys]).astype(np.float32)
    X -= X.mean(axis=0, keepdims=True)
    C = np.cov(X.T)
    w, V = np.linalg.eig(C)
    vmaj = V[:, np.argmax(w)]
    vertical = np.array([0.0, 1.0], dtype=np.float32)
    cosang = abs(np.dot(vmaj / (np.linalg.norm(vmaj) + 1e-9), vertical))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

def tortuosity(mask_bool):
    """Perimeter / diameter (using convex hull extremes as diameter)."""
    m = (mask_bool.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    per = float(cv2.arcLength(c, True))
    pts = c.reshape(-1, 2)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    diam = 1.0
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            d = np.linalg.norm(hull[i] - hull[j])
            if d > diam:
                diam = d
    return per / (diam + 1e-6)

def touches_gate(mask_bool, H):
    """Return True if any pixel of the blob intersects the horizontal gate band."""
    y_center = int(round(GATE_OFFSET_FRAC * H))
    band_h   = int(max(1, round(GATE_BAND_FRAC * H)))
    y1 = max(0, y_center - band_h // 2)
    y2 = min(H, y_center + (band_h - band_h // 2))
    ys, _ = np.where(mask_bool)
    return np.any((ys >= y1) & (ys < y2)), (y1, y2)

def classify_blob(mask_bool, H, W):
    """Decide rim/flow/ignore and return label + features."""
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
    ang  = pca_angle(mask_bool)     # degrees from vertical
    tort = tortuosity(mask_bool)    # perimeter / diameter

    # gate-touch (optional)
    gate_ok, (gy1, gy2) = touches_gate(mask_bool, H)

    # --- FLOW acceptance ---
    flow_cond = (
        (ang <= ANGLE_FLOW_MAX) and
        (y_span >= Y_SPAN_MIN_FLOW) and
        (top_frac <= TOP_FRAC_MAX_FLOW) and
        ((not REQUIRE_GATE_TOUCH) or gate_ok)
    )

    # --- RIM rejection ---
    rim_cond = (
        (ang >= ANGLE_RIM_MIN) or
        ((top_frac >= TOP_DOM_FRAC) and (tort >= TORTUOSITY_RIM)) or
        ((y_span <= Y_SPAN_MAX) and (x_span >= X_SPAN_MIN))
    )

    if flow_cond:
        label = "flow"
    elif rim_cond:
        label = "rim"
    else:
        label = "ignore"

    details = {
        "area": area,
        "top_frac": round(top_frac, 3),
        "angle_deg": round(ang, 1),
        "tortuosity": round(tort, 3),
        "y_span": round(y_span, 3),
        "x_span": round(x_span, 3),
        "gate_touch": bool(gate_ok),
        "gate_y1": int(gy1),
        "gate_y2": int(gy2),
        "bbox": [int(x0), int(y0), int(x1), int(y1)]
    }
    return label, details

# ------------------ main ------------------

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
        red = heat[..., 2].astype(np.uint8)

        # very permissive red mask
        mask = (red > RED_THR).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            # nothing to filter; copy through
            cv2.imwrite(os.path.join(OUT_DIR, Path(fp).name), heat)
            continue

        keep_mask = np.zeros((H, W), np.uint8)  # flow
        rim_mask  = np.zeros((H, W), np.uint8)  # rim/splash

        for lab in range(1, num):
            comp = (labels == lab)
            label, det = classify_blob(comp, H, W)
            det.update({"frame_idx": idx, "blob_id": lab, "label": label})
            rows.append(det)

            if label == "flow":
                keep_mask[comp] = 255
            elif label == "rim":
                rim_mask[comp] = 255
            # ignore → dropped

        # Build output: keep original reds for flow; set rim to background blue
        out = heat.copy()
        # estimate background blue level from non-red pixels
        non_red = heat[..., 0][red <= RED_THR]
        bg_b = int(np.median(non_red)) if non_red.size else 64
        out[rim_mask > 0] = (bg_b, 0, 0)

        # (Optional) visualize the gate band for debugging
        if REQUIRE_GATE_TOUCH:
            y_center = int(round(GATE_OFFSET_FRAC * H))
            band_h   = int(max(1, round(GATE_BAND_FRAC * H)))
            y1 = max(0, y_center - band_h // 2)
            y2 = min(H, y_center + (band_h - band_h // 2))
            out[y1:y2, :, 1] = np.maximum(out[y1:y2, :, 1], 64)  # faint green band

        cv2.imwrite(os.path.join(OUT_DIR, Path(fp).name), out)

    if rows:
        df = pd.DataFrame(rows, columns=[
            "frame_idx","blob_id","label","area","top_frac","angle_deg",
            "tortuosity","y_span","x_span","gate_touch","gate_y1","gate_y2","bbox"
        ])
        df.to_csv(CSV_DECISIONS, index=False)
        print(f"Saved rim/flow decisions → {CSV_DECISIONS}")
        print(f"Filtered frames (rims removed, flows kept) → {OUT_DIR}")
    else:
        print("No blobs found / nothing to log.")

if __name__ == "__main__":
    main()
