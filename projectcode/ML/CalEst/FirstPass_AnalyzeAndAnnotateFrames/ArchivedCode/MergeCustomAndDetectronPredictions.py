#!/usr/bin/env python3
# fuse_detectron_person_with_custom_yolo.py
# Draw Detectron "person" on top of CustomYolo images and write TWO CSVs:
#  1) persons_summary.csv        -> quick per-row summary (your original)
#  2) all_images_summary.csv     -> detailed rows incl. img size + det_idx

from pathlib import Path
import cv2, csv

# ---- PATHS (edit if needed) ----
DETR_TXT_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/DetectronPerson/txt")
CUSTOM_YOLO_IMG_DIR = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/CustomYolo")
RAW_FRAMES_DIR      = Path("/app/mediaFiles/videos/InputVideos/ProteinShake/ExtractFrames")
OUT_DIR             = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/FirstPass_ImageAnnotation/FusedPreview")
CSV_OUT_SUMMARY     = OUT_DIR / "persons_summary.csv"
CSV_OUT_ALL         = OUT_DIR / "all_images_summary.csv"   # <--- added

ACCEPTED_PERSON_IDS = {0, 900}   # COCO person or your 900 id
OUT_DIR.mkdir(parents=True, exist_ok=True)

def yolo_txt_to_xyxy(txt_path, W, H):
    """Parse YOLO txt -> list of (cls, x1,y1,x2,y2, conf)."""
    dets = []
    if not txt_path.exists():
        return dets
    for ln in txt_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
        dets.append((cls, x1, y1, x2, y2, conf))
    return dets

def draw_box(img, x1,y1,x2,y2, label, color=(255,0,255)):
    x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img, label, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def main():
    # CSV headers
    rows_summary = [("frame","num_person","x1","y1","x2","y2","conf","cls")]
    rows_all     = [("frame","img_w","img_h","det_idx","cls","x1","y1","x2","y2","conf")]

    txt_files = sorted(DETR_TXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[WARN] No txt files in {DETR_TXT_DIR}")
        return

    for txt in txt_files:
        stem = txt.stem  # e.g., frame_00063

        # prefer CustomYolo image; else fallback to raw frame
        base_img = (CUSTOM_YOLO_IMG_DIR / f"{stem}.jpg")
        if not base_img.exists():
            base_img = RAW_FRAMES_DIR / f"{stem}.jpg"

        img = cv2.imread(str(base_img))
        if img is None:
            print(f"[MISS] cannot read image for {stem} at {base_img}")
            continue
        H, W = img.shape[:2]

        dets = yolo_txt_to_xyxy(txt, W, H)
        persons = [(cls,x1,y1,x2,y2,conf) for (cls,x1,y1,x2,y2,conf) in dets if cls in ACCEPTED_PERSON_IDS]

        # draw & collect CSV
        for k, (cls,x1,y1,x2,y2,conf) in enumerate(persons):
            draw_box(img, x1,y1,x2,y2, f"person {conf:.2f}")
            rows_summary.append((f"{stem}.jpg", len(persons), f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{conf:.3f}", cls))
            rows_all.append((f"{stem}.jpg", W, H, k, cls, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{conf:.3f}"))

        # if no persons, still emit an all.csv row so you can see the frame
        if not persons:
            rows_all.append((f"{stem}.jpg", W, H, "", "", "", "", "", "", ""))

        out_path = OUT_DIR / f"{stem}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"[OK] {stem}: persons drawn={len(persons)} -> {out_path}")

    # write CSVs
    with open(CSV_OUT_SUMMARY, "w", newline="") as f:
        csv.writer(f).writerows(rows_summary)
    with open(CSV_OUT_ALL, "w", newline="") as f:
        csv.writer(f).writerows(rows_all)

    print(f"[DONE] persons_summary.csv -> {CSV_OUT_SUMMARY}")
    print(f"[DONE] all_images_summary.csv -> {CSV_OUT_ALL}")

if __name__ == "__main__":
    main()
