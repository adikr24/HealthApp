#!/usr/bin/env python3
"""
FSM_HandOnlyClass — contextual hand-only lookback logic (OOI = object only)

Logic:
1) Read all_activity_merged_frames.csv (activity states)
2) Read hand_roi_closest_object.csv (hand-object proximity)
3) For each hand-only segment:
   - Look back 100 frames from start_idx
   - If <10 frames → OOI = "NA"
   - Else → OOI = most frequent nearest_object (e.g., 'almond')
4) Safely MERGE 'OOI' into fsm_heuristics_HandOnly.csv (do not overwrite)
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_PATH  = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
HAND_PATH = BASE / "HandActivity" / "hand_roi_closest_object.csv"
OUT_PATH  = BASE / "FSM" / "fsm_heuristics_HandOnly.csv"

LOOKBACK = 100
MIN_FRAMES = 10

# ---------- Helpers ----------
def frame_to_idx(frame_name: str) -> int:
    m = re.search(r"(\d+)", Path(frame_name).stem)
    return int(m.group(1)) if m else -1

def _ensure_OOI_after_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns so 'OOI' is immediately after 'classes'."""
    if "classes" not in df.columns or "OOI" not in df.columns:
        return df
    cols = list(df.columns)
    cols.remove("OOI")
    insert_pos = cols.index("classes") + 1
    cols.insert(insert_pos, "OOI")
    return df[cols]

# ---------- FSM ----------
class FSM:
    def __init__(self):
        self.act = None
        self.hand = None
        self.OOI_map = {}

    def run(self):
        self._load_inputs()
        self._process_hand_segments()
        self._write_output_safe()

    def _load_inputs(self):
        self.act = pd.read_csv(ACT_PATH)
        self.hand = pd.read_csv(HAND_PATH)

        self.act.columns = [c.strip() for c in self.act.columns]
        self.hand.columns = [c.strip() for c in self.hand.columns]
        self.hand["_frame_idx"] = self.hand["frame"].astype(str).apply(frame_to_idx)

    def _process_hand_segments(self):
        hand_rows = self.act[self.act["classes"].astype(str).str.lower().str.strip() == "hand"]
        if hand_rows.empty:
            print("[OK] No hand-only activities found.")
            return
        else:
            print('entering HandOnly Class')
        for _, row in hand_rows.iterrows():
            start_idx = int(row["start_idx"])
            state_id = int(row["state_id"])

            lookback_start = max(0, start_idx - LOOKBACK)
            win = self.hand[(self.hand["_frame_idx"] >= lookback_start) &
                            (self.hand["_frame_idx"] < start_idx)]

            if len(win) < MIN_FRAMES:
                ooi = "NA"
            else:
                counts = Counter(win["nearest_object"].astype(str))
                ooi, _ = counts.most_common(1)[0]

            self.OOI_map[state_id] = ooi

    def _write_output_safe(self):
        base = self.act.copy()
        base.columns = [c.strip() for c in base.columns]

        ooi_rows = [{"state_id": int(r["state_id"]),
                     "OOI": self.OOI_map.get(int(r["state_id"]), "NA")}
                    for _, r in base.iterrows()]
        ooi_df = pd.DataFrame(ooi_rows)

        if OUT_PATH.exists():
            existing = pd.read_csv(OUT_PATH)
            existing.columns = [c.strip() for c in existing.columns]

            merged = existing.merge(ooi_df, on="state_id", how="left", suffixes=("", "_new"))
            if "OOI_new" in merged.columns:
                merged["OOI"] = merged["OOI_new"].where(merged["OOI_new"].notna(),
                                                        merged.get("OOI", "NA"))
                merged.drop(columns=["OOI_new"], inplace=True)

            merged = _ensure_OOI_after_classes(merged)
            merged.to_csv(OUT_PATH, index=False)
            print(f"[OK] FSM_HandOnlyClass merged 'OOI' (object-only) into: {OUT_PATH}")
        else:
            if "OOI" in base.columns:
                base.drop(columns=["OOI"], inplace=True)
            out_df = base.merge(ooi_df, on="state_id", how="left")
            out_df = _ensure_OOI_after_classes(out_df)
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(OUT_PATH, index=False)
            print(f"[OK] FSM_HandOnlyClass wrote: {OUT_PATH}")

# ---------- Entry ----------
if __name__ == "__main__":
    FSM().run()
