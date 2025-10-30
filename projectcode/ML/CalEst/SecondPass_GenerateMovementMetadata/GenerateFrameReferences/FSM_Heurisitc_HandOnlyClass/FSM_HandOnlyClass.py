#!/usr/bin/env python3
"""
FSM_HandOnlyClass — contextual hand-only lookback logic (safe merge)

Logic:
1) Read all_activity_merged_frames.csv (activity states)
2) Read hand_roi_closest_object.csv (hand-object proximity)
3) For each hand-only segment:
   - Look back 100 frames from start_idx
   - If <10 frames in that window → status = "not enough frames to generate data — moving onwards"
   - Else → status = "previous object most frequent: {object_name} ({count} frames)"
4) Safely MERGE 'status' into fsm_heuristics.csv (do not overwrite). Place 'status' after 'classes'.
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_PATH  = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
HAND_PATH = BASE / "HandActivity" / "hand_roi_closest_object.csv"
OUT_PATH  = BASE / "fsm_heuristics.csv"

# ---------- Helpers ----------
def frame_to_idx(frame_name: str) -> int:
    m = re.search(r"(\d+)", Path(frame_name).stem)
    return int(m.group(1)) if m else -1

def _ensure_status_after_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns so 'status' is immediately after 'classes' (if both exist)."""
    if "classes" not in df.columns or "status" not in df.columns:
        return df
    cols = list(df.columns)
    if "status" in cols:
        cols.remove("status")
    insert_pos = cols.index("classes") + 1
    cols.insert(insert_pos, "status")
    return df[cols]

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
        self._write_output_safe()

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

        for _, row in hand_rows.iterrows():
            start_idx = int(row["start_idx"])
            state_id = int(row["state_id"])

            # 100-frame lookback
            lookback_start = max(0, start_idx - 100)
            win = self.hand[(self.hand["_frame_idx"] >= lookback_start) &
                            (self.hand["_frame_idx"] < start_idx)]

            if len(win) < 10:
                status = "not enough frames to generate data — moving onwards"
            else:
                counts = Counter(win["nearest_object"].astype(str))
                most_common_obj, count = counts.most_common(1)[0]
                status = f"previous object most frequent: {most_common_obj} ({count} frames)"

            self.status_map[state_id] = status

    # ---- safe write/merge ----
    def _write_output_safe(self):
        # Start from activity; build (state_id, status)
        base = self.act.copy()
        base.columns = [c.strip() for c in base.columns]

        status_rows = [{"state_id": int(r["state_id"]),
                        "status": self.status_map.get(int(r["state_id"]), "")}
                       for _, r in base.iterrows()]
        status_df = pd.DataFrame(status_rows)

        if OUT_PATH.exists():
            # Merge into existing fsm_heuristics.csv (preserve other columns like OOI)
            existing = pd.read_csv(OUT_PATH)
            existing.columns = [c.strip() for c in existing.columns]

            merged = existing.merge(status_df, on="state_id", how="left", suffixes=("", "_new"))
            if "status_new" in merged.columns:
                merged["status"] = merged["status_new"].where(merged["status_new"].notna(),
                                                              merged.get("status", ""))
                merged.drop(columns=["status_new"], inplace=True)

            merged = _ensure_status_after_classes(merged)
            merged.to_csv(OUT_PATH, index=False)
            print(f"[OK] FSM_HandOnlyClass merged 'status' into: {OUT_PATH}")
        else:
            # Create fresh fsm_heuristics.csv from activity + status
            if "status" in base.columns:
                base.drop(columns=["status"], inplace=True)
            out_df = base.merge(status_df, on="state_id", how="left")
            out_df = _ensure_status_after_classes(out_df)
            out_df.to_csv(OUT_PATH, index=False)
            print(f"[OK] FSM_HandOnlyClass wrote: {OUT_PATH}")

# ---------- Entry ----------
if __name__ == "__main__":
    FSM().run()
