#!/usr/bin/env python3
"""
FSM_HandOnlyClass — contextual hand-only lookback logic

Logic:
1. Read all_activity_merged_frames.csv (activity states)
2. Read hand_roi_closest_object.csv (hand-object proximity)
3. For each hand-only segment:
   - Determine lookback window (100 frames before start_idx)
   - If <10 frames in that window → status = "not enough frames to generate data — moving onwards"
   - Else → count which 'nearest_object' occurs most often in that window
     → status = "previous object most frequent: {object_name} ({count} frames)"
4. Writes fsm_heuristics.csv with 'status' inserted right after 'classes'
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_PATH = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
HAND_PATH = BASE / "HandActivity" / "hand_roi_closest_object.csv"
OUT_PATH = BASE / "fsm_heuristics.csv"

# ---------- Helpers ----------
def frame_to_idx(frame_name: str) -> int:
    m = re.search(r"(\d+)", Path(frame_name).stem)
    return int(m.group(1)) if m else -1

# ---------- FSM ----------
class FSM:
    def __init__(self):
        self.act = None
        self.hand = None
        self.status_map = {}

    # ---- main driver ----
    def run(self):
        self._load_inputs()
        self._process_hand_segments()
        self._write_output()

    # ---- load ----
    def _load_inputs(self):
        self.act = pd.read_csv(ACT_PATH)
        self.hand = pd.read_csv(HAND_PATH)

        self.act.columns = [c.strip() for c in self.act.columns]
        self.hand.columns = [c.strip() for c in self.hand.columns]
        self.hand["_frame_idx"] = self.hand["frame"].astype(str).apply(frame_to_idx)

    # ---- process ----
    def _process_hand_segments(self):
        hand_rows = self.act[self.act["classes"].astype(str).str.lower().str.strip() == "hand"]
        if hand_rows.empty:
            print("[OK] No hand-only activities found.")
            return

        for idx, row in hand_rows.iterrows():
            start_idx = int(row["start_idx"])
            state_id = int(row["state_id"])

            # determine 100-frame lookback
            lookback_start = max(0, start_idx - 100)
            lookback_window = self.hand[(self.hand["_frame_idx"] >= lookback_start) & (self.hand["_frame_idx"] < start_idx)]

            if len(lookback_window) < 10:
                status = "not enough frames to generate data — moving onwards"
            else:
                # count most frequent nearest_object
                counts = Counter(lookback_window["nearest_object"].astype(str))
                most_common_obj, count = counts.most_common(1)[0]
                status = f"previous object most frequent: {most_common_obj} ({count} frames)"

            self.status_map[state_id] = status

    # ---- write ----
    def _write_output(self):
        out_df = self.act.copy()
        if "status" in out_df.columns:
            out_df.drop(columns=["status"], inplace=True)

        insert_at = list(out_df.columns).index("classes") + 1
        status_list = []

        for _, row in out_df.iterrows():
            sid = int(row["state_id"])
            if sid in self.status_map:
                status_list.append(self.status_map[sid])
            else:
                status_list.append("")

        out_df.insert(insert_at, "status", status_list)
        out_df.to_csv(OUT_PATH, index=False)
        print(f"[OK] FSM_HandOnlyClass wrote: {OUT_PATH}")

# ---------- Entry ----------
if __name__ == "__main__":
    FSM().run()
