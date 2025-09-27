# detectron_relevant_annotator.py
from pathlib import Path
from typing import Optional, Tuple, Set, Iterable, Union
import csv
import cv2
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog


class DetectronRelevantAnnotator:
    """
    Annotate images/frames with Detectron2 detections filtered by a set/CSV of relevant labels.

    Parameters
    ----------
    model_cfg : str
        Detectron2 model zoo config (e.g., "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml").
    score_threshold : float
        Confidence threshold (e.g., 0.5).
    relevant_labels_csv : Optional[str | Path]
        CSV "YoloAndDetectronTrainedClasses,Relevant" — keep rows with Relevant == 'Y'.
    relevant_labels : Optional[Iterable[str]]
        Explicit list of class names to keep (lowercase). If provided, overrides CSV.
    device : Optional[str]
        "cpu" to force CPU; else auto (CUDA if available).
    line_thickness : int
        Rectangle thickness.
    font_scale : float
        Label text size for cv2.putText.
    weights_path : Optional[str | Path]
        Path to custom weights (.pth). If None, uses model zoo weights for model_cfg.
    """

    def __init__(
        self,
        *,
        model_cfg: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        score_threshold: float = 0.5,
        relevant_labels_csv: Optional[Union[str, Path]] = None,
        relevant_labels: Optional[Iterable[str]] = None,
        device: Optional[str] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_threshold)
        cfg.MODEL.WEIGHTS = (
            str(weights_path) if weights_path else model_zoo.get_checkpoint_url(model_cfg)
        )
        if device == "cpu":
            cfg.MODEL.DEVICE = "cpu"  # else auto-select GPU if available

        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # Class mappings
        self.class_names = [str(n).lower() for n in self.metadata.thing_classes]
        self.name_to_id = {n: i for i, n in enumerate(self.class_names)}
        self.id_to_name = {i: n for n, i in self.name_to_id.items()}

        # Relevant labels
        if relevant_labels is not None:
            rel = {str(x).strip().lower() for x in relevant_labels if str(x).strip()}
        elif relevant_labels_csv is not None:
            rel = self._load_relevant_labels_csv(Path(relevant_labels_csv))
        else:
            rel = set(self.class_names)  # default: allow all classes

        self.relevant_ids: Set[int] = {self.name_to_id[n] for n in rel if n in self.name_to_id}

        self.line_thickness = int(line_thickness)
        self.font_scale = float(font_scale)

    # ------------------------ Public API ------------------------

    def process_image(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Annotate a BGR image in-memory. Returns (annotated_img, has_relevant)."""
        return self._annotate_frame(img_bgr)

    def process_image_file(self, image_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """Read → annotate → write a single image. Returns whether any relevant detections were found."""
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
        save_empty: bool = False
    ) -> int:
        """Annotate all matching images in a directory. Returns count saved."""
        frames = sorted(Path(frames_dir).glob(glob_pat))
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
        """Decode a video and save annotated frames as images. Returns count saved."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_idx, saved = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % max(1, every_nth) == 0:
                    annotated, has_rel = self._annotate_frame(frame)
                    if has_rel or save_empty:
                        out_path = out_dir / f"{prefix}_{frame_idx:06d}.jpg"
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
        outputs = self.predictor(img_bgr)
        inst = outputs.get("instances", None)
        annotated = img_bgr.copy()
        has_relevant = False

        if inst is None or len(inst) == 0:
            return annotated, False

        inst = inst.to("cpu")
        if not (hasattr(inst, "pred_boxes") and hasattr(inst, "pred_classes") and hasattr(inst, "scores")):
            return annotated, False

        boxes = inst.pred_boxes.tensor.numpy()
        classes = inst.pred_classes.numpy()
        scores = inst.scores.numpy()

        for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, scores):
            if int(cls_id) not in self.relevant_ids:
                continue
            has_relevant = True

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), self.line_thickness)

            name = self.id_to_name.get(int(cls_id), str(int(cls_id)))
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
        Load labels from CSV where 'Relevant' == 'Y'.
        First column: class name; second column: relevance.
        Example:
            YoloAndDetectronTrainedClasses,Relevant
            person,Y
            cup,N
        """
        if not csv_path.exists():
            return set()

        with open(csv_path, "r", newline="") as f:
            rows = list(csv.reader(f))

        if not rows or len(rows[0]) < 2:
            return set()

        header = [c.strip().lower() for c in rows[0]]
        name_idx = header.index("yoloanddetectrontrainedclasses") if "yoloanddetectrontrainedclasses" in header else 0
        rel_idx = header.index("relevant") if "relevant" in header else 1

        labels: Set[str] = set()
        for row in rows[1:]:
            if not row or len(row) <= max(name_idx, rel_idx):
                continue
            name = row[name_idx].strip().lower()
            rel = row[rel_idx].strip().lower()
            if name and rel == "y":
                labels.add(name)
        return labels
