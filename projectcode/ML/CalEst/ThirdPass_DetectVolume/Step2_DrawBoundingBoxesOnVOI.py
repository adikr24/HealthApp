#!/usr/bin/env python3
# Draws blender mouth ROI using coordinates from voi_bbox_and_hands_labels.csv
# and writes them to flow_rois.csv for downstream pixel-diff analysis.

from pathlib import Path
import csv
import cv2

#### CONFIG ##############################################################
IN_CSV = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/voi_bbox_and_hands_labels.csv")

# Folder containing frames to annotate (where blender frames exist)
FRAMES_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/VOIYolo")

# Output folder for annotated ROI frames
OUT_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/VOIYolo_ROI")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path for CSV to save ROI coordinates
FLOW_ROIS_CSV = OUT_DIR / "flow_rois.csv"

ROI_COLOR = (0, 0, 255)  # red
THICK = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
#######################################################################


def draw_roi_and_export(in_csv: Path, frames_dir: Path, out_dir: Path, csv_out: Path):
    """Draw stored blender mouth ROI from the metadata CSV and export flow_rois.csv."""
    with in_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("cls_name_norm") or "").strip().lower() == "blender"]

    # Prepare CSV for output
    with csv_out.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["filename", "roi_x1", "roi_y1", "roi_x2", "roi_y2"])

        for r in rows:
            fname = Path(r["frame"]).name
            img_path = frames_dir / fname
            if not img_path.exists():
                continue

            rx1 = r.get("roi_x1"); ry1 = r.get("roi_y1")
            rx2 = r.get("roi_x2"); ry2 = r.get("roi_y2")
            if not (rx1 and ry1 and rx2 and ry2):
                continue

            try:
                rx1, ry1, rx2, ry2 = map(int, map(float, [rx1, ry1, rx2, ry2]))
            except ValueError:
                continue

            # Draw ROI on image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), ROI_COLOR, THICK)
            cv2.putText(img, "mouth-ROI", (rx1, max(20, ry1 - 6)),
                        FONT, 0.6, ROI_COLOR, 2, cv2.LINE_AA)

            # Save annotated frame
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), img)
            writer.writerow([fname, rx1, ry1, rx2, ry2])

            print(f"[INFO] ROI drawn and logged for {fname}")

    print(f"[DONE] Annotated frames → {out_dir}")
    print(f"[DONE] ROI CSV written → {csv_out}")


if __name__ == "__main__":
    draw_roi_and_export(IN_CSV, FRAMES_DIR, OUT_DIR, FLOW_ROIS_CSV)
