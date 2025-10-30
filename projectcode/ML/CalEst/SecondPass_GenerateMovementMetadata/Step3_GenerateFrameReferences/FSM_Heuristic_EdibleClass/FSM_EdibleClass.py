#!/usr/bin/env python3
"""
FSM_EdibleClass — identifies edible classes and writes OOI (object of interest).

Logic:
1) Read all_activity_merged_frames.csv (activity segmentation)
2) Read YoloLabels.csv -> edible classes where edible == 'Y'
3) For each activity row:
   - Split 'classes' by '|'
   - If any token ∈ edible_set -> OOI = that token (lowercased)
   - Else OOI = ""
4) Write fsm_heuristics_EdibleClass.csv with 'OOI' inserted after 'classes'
"""

import pandas as pd
from pathlib import Path

# ---------- Paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_PATH = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
YOLO_LABELS = Path("/app/mediaFiles/StaticCSVLists/YoloLabels.csv")
OUT_PATH = BASE / "FSM" / "fsm_heuristics_EdibleClass.csv"


# ---------- FSM ----------
class FSM:
    def __init__(self):
        self.act = None
        self.edible_set = set()

    def run(self):
        print("[DEBUG] Starting FSM_EdibleClass (OOI object-only)")
        self._load_activity()
        self._load_edible_classes()
        self._generate_ooi_column()
        print(f"[OK] FSM_EdibleClass wrote: {OUT_PATH}")
        print("[DEBUG] Done.")

    # ---- Load activity segmentation ----
    def _load_activity(self):
        if not ACT_PATH.exists():
            raise FileNotFoundError(f"Missing activity CSV: {ACT_PATH}")
        self.act = pd.read_csv(ACT_PATH)
        self.act.columns = [c.strip() for c in self.act.columns]
        print(f"[DEBUG] Loaded {len(self.act)} activity rows from {ACT_PATH}")

    # ---- Load edible class list ----
    def _load_edible_classes(self):
        if not YOLO_LABELS.exists():
            raise FileNotFoundError(f"Missing YoloLabels.csv: {YOLO_LABELS}")
        labels = pd.read_csv(YOLO_LABELS)
        labels.columns = [c.strip() for c in labels.columns]
        col_class = next(c for c in labels.columns if c.lower().startswith("yoloanddetectrontrainedclasses"))
        col_edible = next(c for c in labels.columns if c.lower() == "edible")

        self.edible_set = {
            str(cls).strip().lower()
            for cls, edible_flag in zip(labels[col_class], labels[col_edible])
            if str(edible_flag).strip().upper() == "Y"
        }
        print(f"[DEBUG] Found {len(self.edible_set)} edible classes")

    # ---- Core logic ----
    def _generate_ooi_column(self):
        ooi_list = []
        for _, row in self.act.iterrows():
            tokens = [t.strip().lower() for t in str(row.get("classes", "")).split("|") if t.strip()]
            ooi = next((t for t in tokens if t in self.edible_set), "NA")
            ooi_list.append(ooi)

        out_df = self.act.copy()
        if "OOI" in out_df.columns:
            out_df.drop(columns=["OOI"], inplace=True)
        insert_at = list(out_df.columns).index("classes") + 1
        out_df.insert(insert_at, "OOI", ooi_list)

        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUT_PATH, index=False)


# ---------- Entry ----------
if __name__ == "__main__":
    FSM().run()
