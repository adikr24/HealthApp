#!/usr/bin/env python3
"""
Purpose: OCR helper for the hand-bottle FSM heuristic.
This module reads video frames from a specified range, preprocesses them,
and runs EasyOCR to extract text that can later be used to infer the bottle's
contents or object of interest.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Union
import cv2
import easyocr

class FrameOCRRunner:

    def __init__(
        self,
        frames_dir: Union[str, Path],
        start_idx: int,
        end_idx: int,
        langs: Optional[Iterable[str]] = None,
        use_gpu: bool = True,
        do_grayscale: bool = True,
        do_upscale: bool = True,
        target_short_side: int = 1024,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        self.langs = list(langs) if langs else ['en']
        self.use_gpu = bool(use_gpu)
        self.do_grayscale = bool(do_grayscale)
        self.do_upscale = bool(do_upscale)
        self.target_short_side = int(target_short_side)

        # Lazy-initialize the EasyOCR reader
        self._reader: Optional[easyocr.Reader] = None

    # ---------- Public API ----------

    def run(self, print_each: bool = True) -> List[Dict[str, Optional[str]]]:
        reader = self._get_reader()
        results: List[Dict[str, Optional[str]]] = []

        print(f"[INFO] OCR on frames {self.start_idx}–{self.end_idx} in: {self.frames_dir}")

        for idx in range(self.start_idx, self.end_idx + 1):
            fname = self._frame_name(idx)
            fpath = self.frames_dir / fname

            if not fpath.exists():
                if print_each:
                    print(f"[MISS] {fname} not found")
                results.append({"frame": fname, "path": str(fpath), "text": None})
                continue

            img = cv2.imread(str(fpath))
            if img is None:
                if print_each:
                    print(f"[WARN] cannot read {fname}")
                results.append({"frame": fname, "path": str(fpath), "text": None})
                continue

            img_pp = self._preprocess(img)

            try:
                texts = reader.readtext(img_pp, detail=0)  # List[str]
                text = " ".join(t.strip() for t in texts if t and t.strip())
            except Exception as e:
                if print_each:
                    print(f"[ERR ] {fname}: {e}")
                text = None

            if print_each:
                print(f"{fname} → {text if text else '(no text)'}")

            results.append({"frame": fname, "path": str(fpath), "text": text})

        print(f"[OK] Done OCR on frames {self.start_idx}–{self.end_idx}")
        return results

    # ---------- Internals ----------

    def _get_reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self.langs, gpu=self.use_gpu)
        return self._reader

    @staticmethod
    def _frame_name(i: int) -> str:
        return f"frame_{i:05d}.jpg"

    def _preprocess(self, img):
        out = img
        if self.do_grayscale:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if self.do_upscale:
            h, w = out.shape[:2]
            short = min(h, w)
            if 0 < short < self.target_short_side:
                scale = self.target_short_side / short
                nw, nh = int(round(w * scale)), int(round(h * scale))
                out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_CUBIC)
        return out


