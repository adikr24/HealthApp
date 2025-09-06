from __future__ import annotations

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


class InflowStats:
    """
    Combines rim/noise filtering (Step 3) and per-frame quantity estimation (Step 4)
    over ROI heatmaps produced by DetectronCupFlowPipeline.

    Paths are derived from the pipeline `output_dir` under `roi_pixel_metrics/` to
    avoid hardcoded directories. All thresholds and parameters are read from the
    provided JSON config (with safe defaults when absent).
    """

    def __init__(self, config_path: str) -> None:
        if not config_path:
            raise ValueError("config_path is required")
        self.config_path = config_path
        self._load_config(config_path)

        # Derived paths
        self.metrics_root = os.path.join(self.output_dir, "roi_pixel_metrics")
        self.heat_dir = os.path.join(self.metrics_root, "heat_frames")
        self.filtered_dir = os.path.join(self.metrics_root, "rim_filtered_keep_all_flow")
        self.metrics_out_dir = os.path.join(self.metrics_root, "remove_glare")
        os.makedirs(self.filtered_dir, exist_ok=True)
        os.makedirs(self.metrics_out_dir, exist_ok=True)

        # File patterns
        self.heat_name_fmt = self.inflow_heat.get("filename_fmt", "heat_{:05d}.png")
        self.decisions_csv = os.path.join(self.filtered_dir, self.inflow_heat.get("decisions_csv", "rim_decisions.csv"))
        self.metrics_csv = os.path.join(self.metrics_out_dir, self.inflow_score.get("csv_filename", "heat_only_flow_per_frame.csv"))

    # ---------------- Config ----------------
    def _load_config(self, path: str) -> None:
        with open(path, "r") as f:
            cfg = json.load(f)

        def g(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else default

        # roots
        self.output_dir = g(cfg, "output_dir")
        if not self.output_dir:
            raise ValueError("Config must specify 'output_dir'.")

        # heat section (used for scale correction)
        heat = g(cfg, "heat", {})
        self.output_scale = g(heat, "output_scale", 4.0)

        # Inflow parameters (Step3/Step4)
        inflow = g(cfg, "inflow", {})
        self.inflow_heat = g(inflow, "filter", {})  # Step 3
        self.inflow_score = g(inflow, "score", {})  # Step 4

        # Step 3 defaults
        self.red_thr_keep = int(g(self.inflow_heat, "red_thr", 6))
        self.min_pixels = int(g(self.inflow_heat, "min_pixels", 12))
        self.top_band_frac = float(g(self.inflow_heat, "top_band_frac", 0.15))

        self.angle_flow_max = float(g(self.inflow_heat, "angle_flow_max", 30.0))
        self.y_span_min_flow = float(g(self.inflow_heat, "y_span_min_flow", 0.30))
        self.top_frac_max_flow = float(g(self.inflow_heat, "top_frac_max_flow", 0.50))
        self.require_gate_touch = bool(g(self.inflow_heat, "require_gate_touch", True))
        self.gate_offset_frac = float(g(self.inflow_heat, "gate_offset_frac", 0.08))
        self.gate_band_frac = float(g(self.inflow_heat, "gate_band_frac", 0.06))

        self.angle_rim_min = float(g(self.inflow_heat, "angle_rim_min", 55.0))
        self.tortuosity_rim = float(g(self.inflow_heat, "tortuosity_rim", 1.60))
        self.y_span_max = float(g(self.inflow_heat, "y_span_max", 0.25))
        self.x_span_min = float(g(self.inflow_heat, "x_span_min", 0.35))

        # Step 4 defaults
        self.red_thr_score = int(g(self.inflow_score, "red_thr", 12))
        self.top_crop_frac = float(g(self.inflow_score, "top_crop_frac", 0.00))
        self.score_gate_offset_frac = float(g(self.inflow_score, "gate_offset_frac", 0.08))
        self.score_gate_band_frac = float(g(self.inflow_score, "gate_band_frac", 0.06))
        self.cal_ml_per_score = float(g(self.inflow_score, "cal_ml_per_score", 0.0008))
        self.density_g_per_ml = float(g(self.inflow_score, "density_g_per_ml", 1.0))

    # ---------------- Step 3: Remove rim arcs, keep flow ----------------
    @staticmethod
    def _pca_angle(mask_bool: np.ndarray) -> float:
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

    @staticmethod
    def _tortuosity(mask_bool: np.ndarray) -> float:
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

    def _touches_gate(self, mask_bool: np.ndarray, H: int) -> Tuple[bool, Tuple[int, int]]:
        y_center = int(round(self.gate_offset_frac * H))
        band_h = int(max(1, round(self.gate_band_frac * H)))
        y1 = max(0, y_center - band_h // 2)
        y2 = min(H, y_center + (band_h - band_h // 2))
        ys, _ = np.where(mask_bool)
        return np.any((ys >= y1) & (ys < y2)), (y1, y2)

    def _classify_blob(self, mask_bool: np.ndarray, H: int, W: int) -> Tuple[str, Dict[str, object]]:
        area = int(mask_bool.sum())
        if area < self.min_pixels:
            return "ignore", {"area": area}

        ys, xs = np.where(mask_bool)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        y_span = (y1 - y0 + 1) / max(1, H)
        x_span = (x1 - x0 + 1) / max(1, W)

        rim_y = int(round(self.top_band_frac * H))
        top_frac = float((ys < rim_y).sum()) / max(1, area)

        ang = self._pca_angle(mask_bool)
        tort = self._tortuosity(mask_bool)
        gate_ok, (gy1, gy2) = self._touches_gate(mask_bool, H)

        flow_cond = (
            (ang <= self.angle_flow_max) and
            (y_span >= self.y_span_min_flow) and
            (top_frac <= self.top_frac_max_flow) and
            ((not self.require_gate_touch) or gate_ok)
        )

        rim_cond = (
            (ang >= self.angle_rim_min) or
            ((top_frac >= 0.50) and (tort >= self.tortuosity_rim)) or
            ((y_span <= self.y_span_max) and (x_span >= self.x_span_min))
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
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
        }
        return label, details

    def filter_rims_keep_flow(self) -> None:
        rows: List[Dict[str, object]] = []
        paths = sorted(glob.glob(os.path.join(self.heat_dir, "heat_*.png")))
        if not paths:
            print(f"No heat frames found under {self.heat_dir}")
            return

        for p in paths:
            heat = cv2.imread(p, cv2.IMREAD_COLOR)
            if heat is None:
                continue
            try:
                idx = int(Path(p).stem.split("_")[1])
            except Exception:
                continue

            H, W = heat.shape[:2]
            red = heat[..., 2].astype(np.uint8)
            mask = (red > self.red_thr_keep).astype(np.uint8)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num <= 1:
                cv2.imwrite(os.path.join(self.filtered_dir, Path(p).name), heat)
                continue

            keep_mask = np.zeros((H, W), np.uint8)
            rim_mask = np.zeros((H, W), np.uint8)

            for lab in range(1, num):
                comp = (labels == lab)
                label, det = self._classify_blob(comp, H, W)
                det.update({"frame_idx": idx, "blob_id": lab, "label": label})
                rows.append(det)
                if label == "flow":
                    keep_mask[comp] = 255
                elif label == "rim":
                    rim_mask[comp] = 255

            out = heat.copy()
            non_red = heat[..., 0][red <= self.red_thr_keep]
            bg_b = int(np.median(non_red)) if non_red.size else 64
            out[rim_mask > 0] = (bg_b, 0, 0)

            if self.require_gate_touch:
                y_center = int(round(self.gate_offset_frac * H))
                band_h = int(max(1, round(self.gate_band_frac * H)))
                y1 = max(0, y_center - band_h // 2)
                y2 = min(H, y_center + (band_h - band_h // 2))
                out[y1:y2, :, 1] = np.maximum(out[y1:y2, :, 1], 64)

            cv2.imwrite(os.path.join(self.filtered_dir, Path(p).name), out)

        if rows:
            df = pd.DataFrame(rows, columns=[
                "frame_idx", "blob_id", "label", "area", "top_frac", "angle_deg",
                "tortuosity", "y_span", "x_span", "gate_touch", "gate_y1", "gate_y2", "bbox",
            ])
            df.to_csv(self.decisions_csv, index=False)
            print(f"Saved rim/flow decisions → {self.decisions_csv}")
            print(f"Filtered frames (rims removed, flows kept) → {self.filtered_dir}")

    # ---------------- Step 4: Score heat into mL/grams ----------------
    def _frame_score_from_heat(self, heat_bgr: np.ndarray) -> Tuple[float, int, int, int]:
        if heat_bgr is None:
            return 0.0, 0, 0, 0
        H, W = heat_bgr.shape[:2]

        y0 = int(round(self.top_crop_frac * H))
        if y0 >= H:
            return 0.0, 0, 0, 0
        img = heat_bgr[y0:, :, :]
        HH = img.shape[0]

        gate_center = int(round(self.score_gate_offset_frac * HH))
        gate_h = int(max(1, round(self.score_gate_band_frac * HH)))
        gy1 = max(0, gate_center - gate_h // 2)
        gy2 = min(HH, gate_center + (gate_h - gate_h // 2))
        gate_roi = img[gy1:gy2, :, :]

        red = gate_roi[..., 2].astype(np.float32)
        if self.red_thr_score > 0:
            mask = red > self.red_thr_score
            red = red * mask
        raw_sum = red.sum() / 255.0

        scale_correction = 1.0 / float(self.output_scale ** 2)
        score = float(raw_sum * scale_correction)
        used = int((red > 0).sum())
        return score, used, y0 + gy1, y0 + gy2

    def estimate_from_filtered(self) -> None:
        paths = sorted(glob.glob(os.path.join(self.filtered_dir, "heat_*.png")))
        if not paths:
            print(f"No filtered heat frames found under {self.filtered_dir}")
            return

        have_idxs: List[int] = []
        for p in paths:
            try:
                have_idxs.append(int(Path(p).stem.split("_")[1]))
            except Exception:
                pass
        if not have_idxs:
            print("No valid heat frame indices found for scoring")
            return

        rows: List[Dict[str, object]] = []
        cum_score = 0.0
        cum_ml = 0.0
        cum_g = 0.0

        lo = min(have_idxs)
        hi = max(have_idxs)

        for idx in range(lo, hi + 1):
            fp = os.path.join(self.filtered_dir, f"heat_{idx:05d}.png")
            if not os.path.exists(fp):
                continue
            heat = cv2.imread(fp, cv2.IMREAD_COLOR)
            score, n_pix, gy1, gy2 = self._frame_score_from_heat(heat)

            delta_ml = score * self.cal_ml_per_score
            delta_g = delta_ml * self.density_g_per_ml

            cum_score += score
            cum_ml += delta_ml
            cum_g += delta_g

            rows.append({
                "heat_filename": Path(fp).name,
                "index": idx,
                "red_thr": self.red_thr_score,
                "top_crop_frac": self.top_crop_frac,
                "gate_offset_frac": self.score_gate_offset_frac,
                "gate_band_frac": self.score_gate_band_frac,
                "active_pixels": n_pix,
                "flow_score": score,
                "delta_ml": delta_ml,
                "cum_ml": cum_ml,
                "delta_g": delta_g,
                "cum_g": cum_g,
                "gate_y1": gy1,
                "gate_y2": gy2,
            })

        if not rows:
            print("No frames scored.")
            return

        df = pd.DataFrame(rows)
        df.to_csv(self.metrics_csv, index=False)
        print(f"Saved per-frame metrics → {self.metrics_csv}")
        print(f"Total frames: {len(rows)}")
        print(f"Total score (gate, scale-corrected): {df['flow_score'].sum():.2f}")
        print(f"Total mL:    {df['delta_ml'].sum():.2f}  (CAL_ML_PER_SCORE={self.cal_ml_per_score})")
        print(f"Total grams: {df['delta_g'].sum():.2f}  (DENSITY_G_PER_ML={self.density_g_per_ml})")

    # Convenience
    def run(self) -> None:
        self.filter_rims_keep_flow()
        self.estimate_from_filtered()


__all__ = ["InflowStats"]

