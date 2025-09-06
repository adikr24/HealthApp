"""
DetectronCupFlowPipeline
------------------------

This class merges the logic from:
- Step1_RunDetectron2.py (Detectron2 spoon-in-cup trigger + annotated frames)
- Step1_RunYolo.py      (ROI computation, masks, and CSV outputs)

Key behavior:
- Runs Detectron2 on frames to find when a spoon is in a cup.
- Builds a window of frames around trigger indices (before/after), like Step1_RunDetectron2.
- For those selected frames, finds the best cup bbox and computes a flow ROI (YOLO step logic),
  saving clean frames, annotated frames, ROI masks, and a flow_rois.csv.
- Optionally, can generate heatmaps and flow arrows within the ROIs (subset of Step 2 logic),
  without modifying any existing files.

Note: This is a standalone utility class; it does not modify existing scripts.
"""

from __future__ import annotations

import os
import glob
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Detectron2 imports
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


class DetectronCupFlowPipeline:
    """
    Orchestrates Detectron2 detection (spoon-in-cup triggers) + ROI outputs and optional
    heatmap/arrow generation.

    Usage (example):
        pipe = DetectronCupFlowPipeline(
            config_path="projectcode/ML/ActionDetection/Quantity Estimation/detectron_cupflow_config.json"
        )
        pipe.run_detection_and_rois()
        pipe.generate_heat_and_arrows()  # uses start/end from config
    """

    def __init__(self, config_path: str) -> None:
        setup_logger()

        # Load required configuration from JSON
        if not config_path:
            raise ValueError("config_path is required and must point to a JSON config file.")
        self.config_path = config_path
        self._load_config(config_path)

        # Output subdirs
        self.ann_dir = os.path.join(self.output_dir, "annotated_frames")
        self.clean_dir = os.path.join(self.output_dir, "clean_frames")
        self.mask_dir = os.path.join(self.output_dir, "roi_masks")
        for d in [self.output_dir, self.ann_dir, self.clean_dir, self.mask_dir]:
            os.makedirs(d, exist_ok=True)

        # Paths
        self.csv_rois = os.path.join(self.output_dir, "flow_rois.csv")

        # Detectron2 predictor
        self.predictor, self.cfg = self._build_predictor(self.detectron_score_thr)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.class_names = self.metadata.thing_classes
        try:
            self.cup_idx = self.class_names.index(self.cup_name)
            self.spoon_idx = self.class_names.index(self.spoon_name)
        except ValueError:
            raise RuntimeError(
                f"Required classes not found in metadata. Available: {self.class_names}"
            )

    # -------- Detectron2 helpers --------
    def _build_predictor(self, score_thr: float) -> Tuple[DefaultPredictor, object]:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.detectron_config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
        # We default to downloading weights corresponding to the config key unless overridden
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.detectron_weights_key)
        if self.detectron_device in ("cpu", "cuda"):
            cfg.MODEL.DEVICE = self.detectron_device
        return DefaultPredictor(cfg), cfg

    # -------- Config loader --------
    def _load_config(self, path: str) -> None:
        import json
        with open(path, "r") as f:
            cfg = json.load(f)

        # Simple helpers
        def g(obj, key, default=None):
            return obj.get(key, default) if isinstance(obj, dict) else default

        # Top-level (required)
        self.input_dir = g(cfg, "input_dir")
        self.output_dir = g(cfg, "output_dir")
        self.glob_pat = g(cfg, "glob_pat", "frame_*.jpg")
        if not self.input_dir or not self.output_dir:
            raise ValueError("Config must specify 'input_dir' and 'output_dir'.")

        # Trigger window
        trig = g(cfg, "trigger_window", {})
        self.before_frames = g(trig, "before_frames", 12)
        self.after_frames = g(trig, "after_frames", 12)
        self.inside_mode = g(trig, "inside_mode", "center-in")
        self.iou_min = g(trig, "iou_min", 0.10)

        # Detectron
        det = g(cfg, "detectron", {})
        self.detectron_config_file = g(det, "config_file", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.detectron_weights_key = g(det, "weights_key", self.detectron_config_file)
        self.detectron_score_thr = g(det, "score_thr", 0.7)
        self.detectron_device = g(det, "device", None)

        # Classes
        cls = g(cfg, "classes", {})
        self.cup_name = g(cls, "cup_name", "cup")
        self.spoon_name = g(cls, "spoon_name", "spoon")

        # ROI
        roi = g(cfg, "roi", {})
        self.roi_height_frac = g(roi, "height_frac", 0.40)
        self.roi_width_frac = g(roi, "width_frac", 0.60)
        self.roi_right_inset_frac = g(roi, "right_inset_frac", 0.18)
        self.roi_min_w = g(roi, "min_w", 4)
        self.roi_min_h = g(roi, "min_h", 4)

        # Heat/arrow defaults
        heat = g(cfg, "heat", {})
        self.heat_polarity = g(heat, "polarity", "both")
        self.heat_diff_thr_bright_hi = g(heat, "diff_thr_bright_hi", 12)
        self.heat_diff_thr_dark_hi = g(heat, "diff_thr_dark_hi", 8)
        self.heat_diff_thr_bright_lo = g(heat, "diff_thr_bright_lo", 6)
        self.heat_diff_thr_dark_lo = g(heat, "diff_thr_dark_lo", 4)
        self.heat_mag_thr = g(heat, "mag_thr", 0.9)
        self.heat_vy_thr = g(heat, "vy_thr", 0.4)
        self.heat_vert_ratio = g(heat, "vert_ratio", 2.0)
        self.heat_arrow_step = g(heat, "arrow_step", 8)
        self.heat_arrow_scale = g(heat, "arrow_scale", 2)
        self.heat_output_scale = g(heat, "output_scale", 4.0)
        self.heat_top_crop_frac = g(heat, "top_crop_frac", 0.05)
        self.heat_start_idx = g(heat, "start_idx", None)
        self.heat_end_idx = g(heat, "end_idx", None)

    @staticmethod
    def _bbox_center(b: Sequence[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _center_in(self, b_spoon: Sequence[float], b_cup: Sequence[float]) -> bool:
        cx, cy = self._bbox_center(b_spoon)
        x1, y1, x2, y2 = b_cup
        return (x1 <= cx <= x2) and (y1 <= cy <= y2)

    @staticmethod
    def _iou(a: Sequence[float], b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _spoon_in_cup(self, spoon_boxes: np.ndarray, cup_boxes: np.ndarray) -> bool:
        if len(spoon_boxes) == 0 or len(cup_boxes) == 0:
            return False
        for s in spoon_boxes:
            for c in cup_boxes:
                if self.inside_mode == "center-in":
                    if self._center_in(s, c):
                        return True
                else:  # "iou"
                    if self._iou(s, c) >= self.iou_min:
                        return True
        return False

    # -------- ROI helper (ported from YOLO step) --------
    def compute_flow_roi(
        self,
        cup_xyxy: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        roi_height_frac: Optional[float] = None,
        roi_width_frac: Optional[float] = None,
        right_inset_frac: Optional[float] = None,
        rim_gap_frac_unused: float = 0.03,
        min_roi_w: Optional[int] = None,
        min_roi_h: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Replicates Step1_RunYolo.compute_flow_roi but using a Detectron2 cup box."""
        x1, y1, x2, y2 = cup_xyxy
        w = x2 - x1
        h = y2 - y1

        # Defaults from instance settings if not provided
        if roi_height_frac is None:
            roi_height_frac = self.roi_height_frac
        if roi_width_frac is None:
            roi_width_frac = self.roi_width_frac
        if right_inset_frac is None:
            right_inset_frac = self.roi_right_inset_frac
        if min_roi_w is None:
            min_roi_w = self.roi_min_w
        if min_roi_h is None:
            min_roi_h = self.roi_min_h

        roi_h = int(max(min_roi_h, roi_height_frac * h))
        roi_w = int(max(min_roi_w, roi_width_frac * w))

        cx = x1 + w / 2.0
        rx1 = int(cx - roi_w / 2.0)
        rx2 = rx1 + roi_w

        trim = int(right_inset_frac * w)
        rx2 = rx2 - trim
        if rx2 - rx1 < min_roi_w:
            rx1 = rx2 - min_roi_w

        y_top = int(y1)
        y_bottom = y_top + roi_h

        rx1 = max(0, min(rx1, img_w - 1))
        rx2 = max(0, min(rx2, img_w - 1))
        y_top = max(0, min(y_top, img_h - 1))
        y_bottom = max(0, min(y_bottom, img_h - 1))

        if rx2 <= rx1 or y_bottom <= y_top:
            return None
        return (rx1, y_top, rx2, y_bottom)

    # -------- Phase 1+2: detect triggers, then save selected frames with ROIs --------
    def run_detection_and_rois(self) -> None:
        frames = sorted(glob.glob(os.path.join(self.input_dir, self.glob_pat)))
        if not frames:
            print(f"No frames found under {self.input_dir}/{self.glob_pat}")
            return

        # Pass 1: spoon-in-cup triggers
        trig: List[bool] = [False] * len(frames)
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                continue
            outputs = self.predictor(img)
            inst = outputs["instances"].to("cpu")

            keep = inst.scores >= self.detectron_score_thr
            if keep.sum() == 0:
                continue

            boxes = inst.pred_boxes.tensor.numpy()
            clses = inst.pred_classes.numpy()
            spoon_boxes = boxes[clses == self.spoon_idx] if (clses == self.spoon_idx).any() else []
            cup_boxes = boxes[clses == self.cup_idx] if (clses == self.cup_idx).any() else []

            if len(spoon_boxes) and len(cup_boxes):
                if self._spoon_in_cup(spoon_boxes, cup_boxes):
                    trig[i] = True

        # Build index window around triggers
        to_save = set()
        for i, is_trig in enumerate(trig):
            if is_trig:
                a = max(0, i - self.before_frames)
                b = min(len(frames) - 1, i + self.after_frames)
                for j in range(a, b + 1):
                    to_save.add(j)

        print(f"Triggered frames: {sum(trig)}")
        print(f"Frames to save (with windows): {len(to_save)}")

        # Prepare CSV
        with open(self.csv_rois, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "filename",
                "cup_x1", "cup_y1", "cup_x2", "cup_y2", "cup_conf",
                "roi_x1", "roi_y1", "roi_x2", "roi_y2",
            ])

            # Pass 2: for selected frames, draw annotations + compute ROI
            for i in sorted(to_save):
                fp = frames[i]
                img = cv2.imread(fp)
                if img is None:
                    print(f"Skip unreadable: {fp}")
                    continue
                H, W = img.shape[:2]

                clean = img.copy()

                # Detectron inference
                outputs = self.predictor(img)
                instances = outputs["instances"].to("cpu")

                # Choose best cup by score over threshold
                best = None  # (conf, (x1,y1,x2,y2))
                if hasattr(instances, "pred_boxes"):
                    boxes = instances.pred_boxes.tensor.numpy()
                    clses = instances.pred_classes.numpy()
                    scores = instances.scores.numpy()
                    for b, c, s in zip(boxes, clses, scores):
                        if int(c) == self.cup_idx and float(s) >= self.detectron_score_thr:
                            x1, y1, x2, y2 = [int(v) for v in b.tolist()]
                            cand = (float(s), (x1, y1, x2, y2))
                            if (best is None) or (cand[0] > best[0]):
                                best = cand

                # Save clean frame always
                cv2.imwrite(os.path.join(self.clean_dir, Path(fp).name), clean)

                # Build ROI mask if possible
                mask = np.zeros((H, W), np.uint8)
                if best is not None:
                    cup_conf, cup_xyxy = best
                    flow_roi = self.compute_flow_roi(cup_xyxy, W, H)
                else:
                    flow_roi = None

                if flow_roi is not None:
                    rx1, ry1, rx2, ry2 = flow_roi
                    cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
                cv2.imwrite(os.path.join(self.mask_dir, Path(fp).stem + "_mask.png"), mask)

                # Annotated overlay: Detectron predictions + ROI box
                ann = img.copy()
                try:
                    # Use cv2.cvtColor instead of negative strides to ensure contiguity
                    rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                    v = Visualizer(rgb, metadata=self.metadata, scale=1.0)
                    v = v.draw_instance_predictions(instances)
                    ann = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)
                except Exception:
                    # fallback: no visualizer overlay
                    pass
                if best is not None and flow_roi is not None:
                    x1, y1, x2, y2 = best[1]
                    # Guarantee OpenCV gets a contiguous, writable buffer
                    ann = np.ascontiguousarray(ann)
                    cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    rx1, ry1, rx2, ry2 = flow_roi
                    cv2.rectangle(ann, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                    cv2.putText(
                        ann,
                        f"flow-ROI (-18% right)",
                        (rx1, max(0, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(os.path.join(self.ann_dir, Path(fp).name), ann)

                # CSV row
                if best is None:
                    writer.writerow([Path(fp).name, "", "", "", "", "", "", "", "", ""])
                else:
                    cup_conf, (x1, y1, x2, y2) = best
                    if flow_roi is None:
                        writer.writerow([Path(fp).name, x1, y1, x2, y2, f"{cup_conf:.3f}", "", "", "", ""])
                    else:
                        rx1, ry1, rx2, ry2 = flow_roi
                        writer.writerow([Path(fp).name, x1, y1, x2, y2, f"{cup_conf:.3f}", rx1, ry1, rx2, ry2])

        print(f"Clean frames → {self.clean_dir}")
        print(f"Annotated frames → {self.ann_dir}")
        print(f"ROI masks → {self.mask_dir}")
        print(f"CSV with ROIs → {self.csv_rois}")

    # -------- Phase 3 (optional): generate heatmaps and arrow overlays within ROIs --------
    # A compact adaptation from Step2_DrawHeatMapForCoffee, focused on ROI-only outputs.
    @staticmethod
    def _gamma_LUT(gamma: float = 0.7) -> np.ndarray:
        x = np.arange(256, dtype=np.float32) / 255.0
        y = np.power(x, gamma)
        return (np.clip(y * 255, 0, 255)).astype(np.uint8)

    @staticmethod
    def _build_blue_to_red_colormap() -> np.ndarray:
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            b = int(round(255 * (1.0 - t)))
            g = 0
            r = int(round(255 * t))
            lut[i, 0, :] = (b, g, r)
        return lut

    def generate_heat_and_arrows(self) -> None:
        """Generate ROI heat maps and arrow overlays from clean frames + CSV ROIs."""
        out_dir = os.path.join(self.output_dir, "roi_pixel_metrics")
        heat_dir = os.path.join(out_dir, "heat_frames")
        arrows_dir = os.path.join(out_dir, "arrow_overlays")
        os.makedirs(heat_dir, exist_ok=True)
        os.makedirs(arrows_dir, exist_ok=True)

        # Load ROI CSV
        roi_map: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
        if not os.path.exists(self.csv_rois):
            print(f"Missing ROI CSV: {self.csv_rois}")
            return
        with open(self.csv_rois, "r", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                fn = r.get("filename", "")
                if not fn:
                    continue
                x1 = r.get("roi_x1", ""); y1 = r.get("roi_y1", "")
                x2 = r.get("roi_x2", ""); y2 = r.get("roi_y2", "")
                if (x1 == "" or y1 == "" or x2 == "" or y2 == ""):
                    roi_map[fn] = None
                else:
                    roi_map[fn] = (int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))

        # List frame indices from clean frames
        paths = sorted(glob.glob(os.path.join(self.clean_dir, self.glob_pat)))
        have_idxs: List[int] = []
        for p in paths:
            try:
                idx = int(Path(p).stem.split("_")[1])
                have_idxs.append(idx)
            except Exception:
                pass
        if not have_idxs:
            print(f"No clean frames found under {self.clean_dir}")
            return

        lo = have_idxs[0] if self.heat_start_idx is None else self.heat_start_idx
        hi = have_idxs[-1] if self.heat_end_idx is None else self.heat_end_idx

        # Utility fns
        GAMMA = self._gamma_LUT(0.7)
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        BLUE2RED_LUT = self._build_blue_to_red_colormap()

        def to_gray(img_bgr: np.ndarray) -> np.ndarray:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
            L = lab[..., 0]
            L = CLAHE.apply(L)
            return cv2.LUT(L, GAMMA)

        def hysteresis_mask(diff_pos_u16: np.ndarray, thr_hi: int, thr_lo: int) -> np.ndarray:
            seeds = (diff_pos_u16 >= thr_hi).astype(np.uint8) * 255
            low = (diff_pos_u16 >= thr_lo).astype(np.uint8) * 255
            if seeds.max() == 0:
                return np.zeros_like(seeds, dtype=np.uint8)
            grown = seeds.copy()
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            for _ in range(2):
                dil = cv2.dilate(grown, k, iterations=1)
                grown = cv2.bitwise_and(dil, low)
                grown = cv2.bitwise_or(grown, seeds)
            grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), 1)
            return grown

        def apply_blue2red(gray_0to255: np.ndarray) -> np.ndarray:
            gray_u8 = np.clip(gray_0to255, 0, 255).astype(np.uint8)
            return cv2.LUT(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), BLUE2RED_LUT)

        # Fill defaults from instance settings if not provided
        polarity = self.heat_polarity
        diff_thr_bright_hi = self.heat_diff_thr_bright_hi
        diff_thr_dark_hi = self.heat_diff_thr_dark_hi
        diff_thr_bright_lo = self.heat_diff_thr_bright_lo
        diff_thr_dark_lo = self.heat_diff_thr_dark_lo
        mag_thr = self.heat_mag_thr
        vy_thr = self.heat_vy_thr
        vert_ratio = self.heat_vert_ratio
        arrow_step = self.heat_arrow_step
        arrow_scale = self.heat_arrow_scale
        output_scale = self.heat_output_scale
        top_crop_frac = self.heat_top_crop_frac

        def analyze_pair_roi(A_roi: np.ndarray, B_roi: np.ndarray):
            visA = A_roi.copy(); visB = B_roi.copy()
            if top_crop_frac > 0:
                cut = int(top_crop_frac * visA.shape[0])
                visA[:cut, :] = 0; visB[:cut, :] = 0

            gA = to_gray(visA); gB = to_gray(visB)
            d_bright = (gB.astype(np.int16) - gA.astype(np.int16))
            d_dark = (gA.astype(np.int16) - gB.astype(np.int16))
            pb = np.clip(d_bright, 0, None).astype(np.uint16)
            pd = np.clip(d_dark, 0, None).astype(np.uint16)

            if polarity == "bright":
                seeds = hysteresis_mask(pb, diff_thr_bright_hi, diff_thr_bright_lo)
                pos_val = pb
            elif polarity == "dark":
                seeds = hysteresis_mask(pd, diff_thr_dark_hi, diff_thr_dark_lo)
                pos_val = pd
            else:
                mb = hysteresis_mask(pb, diff_thr_bright_hi, diff_thr_bright_lo)
                md = hysteresis_mask(pd, diff_thr_dark_hi, diff_thr_dark_lo)
                seeds = cv2.bitwise_or(mb, md)
                pos_val = np.maximum(pb, pd)

            pos_val = (pos_val * (seeds > 0)).astype(np.float32)
            if pos_val.max() > 0:
                heat_u8 = (pos_val / (pos_val.max() + 1e-6) * 255).astype(np.uint8)
            else:
                heat_u8 = np.zeros_like(gB, dtype=np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                gA, gB, None,
                pyr_scale=0.5, levels=2, winsize=11, iterations=2,
                poly_n=5, poly_sigma=1.1, flags=0,
            )
            fx, fy = flow[..., 0], flow[..., 1]
            mag = np.sqrt(fx * fx + fy * fy)
            strong = mag > mag_thr
            downward = fy > vy_thr
            vertical = np.abs(fy) >= vert_ratio * (np.abs(fx) + 1e-6)
            down_mask = (strong & downward & vertical)

            heat_u8[seeds == 0] = 0
            return heat_u8, down_mask, fx, fy

        def draw_arrows_on_image(img_full: np.ndarray, roi_box: Tuple[int, int, int, int], down_mask, fx, fy):
            rx1, ry1, rx2, ry2 = roi_box
            sub_h, sub_w = ry2 - ry1, rx2 - rx1
            for y in range(arrow_step // 2, sub_h, arrow_step):
                for x in range(arrow_step // 2, sub_w, arrow_step):
                    if down_mask[y, x]:
                        vx = fx[y, x]; vy = fy[y, x]
                        pt1 = (rx1 + x, ry1 + y)
                        pt2 = (rx1 + int(x + arrow_scale * vx), ry1 + int(y + arrow_scale * vy))
                        cv2.arrowedLine(img_full, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

        # Iterate consecutive frame pairs
        for k in range(lo, hi):
            fnA = f"frame_{k:05d}.jpg"
            fnB = f"frame_{k+1:05d}.jpg"
            fpA = os.path.join(self.clean_dir, fnA)
            fpB = os.path.join(self.clean_dir, fnB)
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

            # Heat (ROI-only), scaled
            heat_roi_bgr = apply_blue2red(heat_u8)
            heat_vis = cv2.resize(heat_roi_bgr, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_NEAREST)
            heat_path = os.path.join(heat_dir, f"heat_{k+1:05d}.png")
            cv2.imwrite(heat_path, heat_vis)

            # Arrow overlay (ROI-only), scaled
            arrow_roi = B[ry1:ry2, rx1:rx2].copy()
            draw_arrows_on_image(arrow_roi, (0, 0, arrow_roi.shape[1], arrow_roi.shape[0]), down_mask, fx, fy)
            arrow_vis = cv2.resize(arrow_roi, None, fx=output_scale, fy=output_scale, interpolation=cv2.INTER_NEAREST)
            arrows_path = os.path.join(arrows_dir, f"arrows_{k+1:05d}.png")
            cv2.imwrite(arrows_path, arrow_vis)

        print(f"Heat frames → {heat_dir}")
        print(f"Arrow overlays → {arrows_dir}")


__all__ = ["DetectronCupFlowPipeline"]
