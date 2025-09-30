# yolo_relevant_annotator.py
from pathlib import Path
from typing import Optional, Tuple, Set, Iterable, Union
import csv
import cv2
import numpy as np
from ultralytics import YOLO


class YoloRelevantAnnotator:
    """
    Annotate images/frames with YOLOv8 detections filtered by a set/CSV of relevant labels.

    Parameters
    ----------
    model_weights : str | Path
        Path to a YOLOv8 .pt weights file (e.g., '/app/.../yolo8l.pt').
    conf_threshold : float
        Confidence threshold for filtering detections (e.g., 0.5).
    relevant_labels_csv : Optional[str | Path]
        CSV file listing relevant labels (one per line, or header 'label').
    relevant_labels : Optional[Iterable[str]]
        Labels to keep (lowercased, e.g., {'cup','bottle'}). If provided, CSV is ignored.
    device : Optional[str]
        'cpu', 'cuda:0', or None to let Ultralytics auto-select.
    line_thickness : int
        Rectangle thickness in pixels.
    font_scale : float
        Label text size for cv2.putText.
    """

    def __init__(
        self,
        *,
        model_weights: Union[str, Path],
        conf_threshold: float,
        relevant_labels_csv: Optional[Union[str, Path]] = None,
        relevant_labels: Optional[Iterable[str]] = None,
        device: Optional[str] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        self.model = YOLO(str(model_weights))
        if device:
            self.model.to(device)

        self.conf_threshold = float(conf_threshold)
        self.line_thickness = int(line_thickness)
        self.font_scale = float(font_scale)

        # Model class maps
        names = getattr(self.model, "names", None) or getattr(self.model.model, "names", {})
        if isinstance(names, dict):
            self.id_to_name = {int(i): str(n) for i, n in names.items()}
        else:
            self.id_to_name = {i: str(n) for i, n in enumerate(names)}
        self.name_to_id = {n.lower(): i for i, n in self.id_to_name.items()}

        # Relevant labels
        if relevant_labels is not None:
            rel = {str(x).strip().lower() for x in relevant_labels if str(x).strip()}
        elif relevant_labels_csv is not None:
            rel = self._load_relevant_labels_csv(Path(relevant_labels_csv))
        else:
            rel = set()  # empty means nothing will be drawn

        # Map to class IDs that exist in this model
        self.relevant_ids: Set[int] = {self.name_to_id[n] for n in rel if n in self.name_to_id}
        if not self.relevant_ids and rel:
            # User provided labels but none matched model classes
            sample = list(self.id_to_name.values())[:10]
            print("[WARN] None of the provided relevant labels matched the model classes.")
            print("       Example model classes:", sample)

    # ------------------------ Public API ------------------------

    def process_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Annotate a single BGR image array in-memory.

        Returns
        -------
        annotated_bgr : np.ndarray
        has_relevant : bool
        """
        return self._annotate_frame(img_bgr)

    def process_image_file(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Read → annotate → write a single image file.

        Returns
        -------
        has_relevant : bool
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        annotated, has_rel = self._annotate_frame(img)
        cv2.imwrite(str(output_path), annotated)
        return has_rel

    def process_frames_dir(
        self,
        *,
        frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
        glob_pat: str = "*.jpg",
        save_empty: bool = False,
        start_index: int = 0,          # <-- NEW: where to start in the sorted list
        max_frames: Optional[int] = None  # <-- NEW: how many frames to process
    ) -> int:
        """
        Annotate all images matching a pattern in a directory.

        Parameters
        ----------
        frames_dir : str | Path
            Directory containing frames.
        output_dir : str | Path
            Where annotated images will be written.
        glob_pat : str
            Glob for frames (e.g., '*.jpg' or 'frame_*.jpg').
        save_empty : bool
            If False, skip frames with no relevant detections.
        start_index : int
            Start at this index in the sorted frame list (default 0).
        max_frames : Optional[int]
            If set, limit to this many frames from start_index.

        Returns
        -------
        saved_count : int
        """
        all_frames = sorted(Path(frames_dir).glob(glob_pat))
        if start_index < 0:
            start_index = 0
        if max_frames is None:
            frames = all_frames[start_index:]
        else:
            frames = all_frames[start_index:start_index + max_frames]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            annotated, has_rel = self._annotate_frame(img)
            if has_rel or save_empty:
                cv2.imwrite(str(out_dir / fp.name), annotated)
                saved += 1
        return saved

    def process_video(
        self,
        *,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        every_nth: int = 1,
        max_frames: Optional[int] = None,
        save_empty: bool = False,
        prefix: str = "frame"
    ) -> int:
        """
        Decode a video and save annotated frames as images.

        Returns
        -------
        saved_count : int
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        frame_idx, saved = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % max(1, every_nth) == 0:
                    annotated, has_rel = self._annotate_frame(frame)
                    if has_rel or save_empty:
                        out_path = Path(output_dir) / f"{prefix}_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(out_path), annotated)
                        saved += 1
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
        return saved

    # ------------------------ Internal helpers ------------------------

    def _annotate_frame(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        res = self.model.predict(img_bgr, conf=self.conf_threshold, verbose=False)[0]
        annotated = img_bgr.copy()
        has_relevant = False

        if not hasattr(res, "boxes") or res.boxes is None:
            return annotated, False

        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            if cls_id not in self.relevant_ids:
                continue

            has_relevant = True
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), self.line_thickness)
            # Label
            name = self.id_to_name.get(cls_id, str(cls_id))
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            y_text = max(y1 - 4, th + 2)
            cv2.rectangle(annotated, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated, has_relevant

    @staticmethod
    def _load_relevant_labels_csv(csv_path: Path) -> Set[str]:
        """
        Load labels from CSV where a column 'Relevant' == 'Y'.
        Assumes first column is class name and second column is relevance.
        Example:
            YoloAndDetectronTrainedClasses,Relevant
            person,Y
            cup,N
        """
        if not csv_path.exists():
            return set()

        labels: Set[str] = set()
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))

        if not rows or len(rows[0]) < 2:
            return set()

        header = [c.strip().lower() for c in rows[0]]
        # try to auto-detect column indices
        name_idx = 0
        rel_idx = 1
        if "yoloanddetectrontrainedclasses".lower() in header:
            name_idx = header.index("yoloanddetectrontrainedclasses")
        if "relevant" in header:
            rel_idx = header.index("relevant")

        for row in rows[1:]:  # skip header
            if not row or len(row) <= max(name_idx, rel_idx):
                continue
            name = row[name_idx].strip().lower()
            rel  = row[rel_idx].strip().lower()
            if name and rel == "y":
                labels.add(name)

        return labels
