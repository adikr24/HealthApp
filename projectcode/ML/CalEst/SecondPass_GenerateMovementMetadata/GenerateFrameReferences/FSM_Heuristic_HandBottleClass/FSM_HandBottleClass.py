#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.GenerateFrameReferences.FSM_Heuristic_HandBottleClass.LabelDetection import FrameOCRRunner

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
CSV_ACTIVITY = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
FRAMES_ROOT  = Path("/app/mediaFiles/videos/InputVideos/ProteinShakeII/ExtractFrames/ProteinShake")
CSV_OUT = BASE / "bottle_ocr_image.csv"  # simplified output file
# ---------- Helpers ----------
def _idx_from_frame(frame_name: str) -> int:
    m = re.search(r"(\d+)", Path(frame_name).stem)
    return int(m.group(1)) if m else -1

def _split_classes(s: str):
    return [t.strip().lower() for t in str(s).split("|") if t.strip()]

def _has_all(tokens, required):
    st = set(tokens)
    return all(r in st for r in required)

# ---------- FSM ----------
class FSM:
    def __init__(self):
        pass

    def run(self):
        act = pd.read_csv(CSV_ACTIVITY)
        act.columns = [c.strip() for c in act.columns]
        all_rows = []
        for _, row in act.iterrows():
            classes = str(row["classes"])
            toks = _split_classes(classes)

            # Trigger on hand+bottle or hand+scoop
            if not (_has_all(toks, ["hand", "bottle"]) or _has_all(toks, ["hand", "scoop"])):
                continue

            start_idx = int(row["start_idx"])

            # Decide lookback window
            if _has_all(toks, ["hand", "scoop"]):
                lookback = 200
                mode = "hand+scoop"
            else:
                lookback = 50
                mode = "hand+bottle"

            win_start = max(0, start_idx - lookback)
            win_end = start_idx - 1

            print(f"\n[INFO] State {int(row['state_id'])} | {classes} | Mode={mode} | Frames {win_start}–{win_end}")

            # Run OCR using your existing runner
            ocr_runner = FrameOCRRunner(
                frames_dir=FRAMES_ROOT,
                start_idx=win_start,
                end_idx=win_end,
                langs=['en'],
                use_gpu=True,
                do_grayscale=True,
                do_upscale=True,
                target_short_side=1024,
            )
            results = ocr_runner.run(print_each=False)

            for r in results:
                fname = r["frame"]
                text = r["text"] or ""
                print(f"  {fname} → {text}")
                all_rows.append({
                    "state_id": int(row["state_id"]),
                    "classes": classes,
                    "frame": fname,
                    "ocr_text": text
                })
                print(f"  {fname} → {text}")

        print("\n[OK] Finished OCR preview for hand+bottle and hand+scoop states.")
         # ----- Write output -----
        out_df = pd.DataFrame(all_rows)
        if out_df.empty:
            print("[WARN] No OCR text extracted — nothing to save.")
            return

        if CSV_OUT.exists():
            prev = pd.read_csv(CSV_OUT)
            merged = pd.concat([prev, out_df], ignore_index=True)
            merged.drop_duplicates(subset=["state_id", "frame"], keep="last", inplace=True)
            merged.to_csv(CSV_OUT, index=False)
        else:
            out_df.to_csv(CSV_OUT, index=False)

        print(f"\n[OK] OCR results written to: {CSV_OUT}")



# ---------- Run directly ----------
if __name__ == "__main__":
    FSM().run()
