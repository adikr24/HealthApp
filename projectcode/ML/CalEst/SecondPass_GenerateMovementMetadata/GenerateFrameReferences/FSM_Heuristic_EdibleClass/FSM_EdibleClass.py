#!/usr/bin/env python3
"""
FSM_EdibleClass
---------------
Marks activity states containing edible classes as 'no action required'.

Logic:
1. Read all_activity_merged_frames.csv (activity segmentation)
2. Read YoloLabels.csv to get all edible classes (where edible == 'Y')
3. For each row in activities:
   - Split 'classes' by '|'
   - If ANY class is in the edible set -> status = "edible object detected — no action required"
   - Else -> status = ""
4. Write fsm_heuristics.csv (same structure, 'status' inserted after 'classes')
"""

import pandas as pd
from pathlib import Path

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_PATH = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
YOLO_LABELS = Path("/app/mediaFiles/StaticCSVLists/YoloLabels.csv")
OUT_PATH = BASE / "fsm_heuristics.csv"

# ---------- FSM ----------
class FSM:
    def __init__(self):
        self.act = None
        self.edible_set = set()

    def run(self):
        print("[DEBUG] Starting FSM_EdibleClass")

        # Load activities
        self.act = pd.read_csv(ACT_PATH)
        self.act.columns = [c.strip() for c in self.act.columns]
        print(f"[DEBUG] Loaded {len(self.act)} activity rows from {ACT_PATH}")

        # Load edible class list
        labels = pd.read_csv(YOLO_LABELS)
        labels.columns = [c.strip() for c in labels.columns]
        col_class = next(c for c in labels.columns if c.lower().startswith("yoloanddetectrontrainedclasses"))
        col_edible = next(c for c in labels.columns if c.lower() == "edible")

        self.edible_set = {
            str(cls).strip().lower()
            for cls, edible_flag in zip(labels[col_class], labels[col_edible])
            if str(edible_flag).strip().upper() == "Y"
        }
        print(f"[DEBUG] Found {len(self.edible_set)} edible classes in YoloLabels.csv")

        # Generate status column
        status_list = []
        for _, row in self.act.iterrows():
            classes = [c.strip().lower() for c in str(row["classes"]).split("|") if c.strip()]
            if any(c in self.edible_set for c in classes):
                status = "edible object detected — no action required"
            else:
                status = ""
            status_list.append(status)

        # Insert 'status' column
        out_df = self.act.copy()
        if "status" in out_df.columns:
            out_df.drop(columns=["status"], inplace=True)

        insert_at = list(out_df.columns).index("classes") + 1
        out_df.insert(insert_at, "status", status_list)
        out_df.to_csv(OUT_PATH, index=False)

        print(f"[OK] FSM_EdibleClass wrote: {OUT_PATH}")
        print("[DEBUG] FSM_EdibleClass finished successfully.")

# ---------- Entry ----------
if __name__ == "__main__":
    FSM().run()
