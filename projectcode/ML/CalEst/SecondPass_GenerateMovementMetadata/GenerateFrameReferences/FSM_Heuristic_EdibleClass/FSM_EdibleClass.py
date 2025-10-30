#!/usr/bin/env python3
"""
FSM_EdibleClass — writes edible OOI (object-only)

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

def main():
    print("[DEBUG] Starting FSM_EdibleClass (OOI object-only)")

    # Load activities
    act = pd.read_csv(ACT_PATH)
    act.columns = [c.strip() for c in act.columns]
    print(f"[DEBUG] Loaded {len(act)} activity rows from {ACT_PATH}")

    # Load edible classes
    labels = pd.read_csv(YOLO_LABELS)
    labels.columns = [c.strip() for c in labels.columns]
    col_class = next(c for c in labels.columns if c.lower().startswith("yoloanddetectrontrainedclasses"))
    col_edible = next(c for c in labels.columns if c.lower() == "edible")

    edible_set = {
        str(cls).strip().lower()
        for cls, edible_flag in zip(labels[col_class], labels[col_edible])
        if str(edible_flag).strip().upper() == "Y"
    }
    print(f"[DEBUG] Found {len(edible_set)} edible classes")

    # Build OOI column
    ooi_list = []
    for _, row in act.iterrows():
        tokens = [t.strip().lower() for t in str(row.get("classes", "")).split("|") if t.strip()]
        ooi = ""
        for t in tokens:
            if t in edible_set:
                ooi = t
                break
        ooi_list.append(ooi)

    # Insert 'OOI' right after 'classes'
    out_df = act.copy()
    if "OOI" in out_df.columns:
        out_df.drop(columns=["OOI"], inplace=True)
    insert_at = list(out_df.columns).index("classes") + 1
    out_df.insert(insert_at, "OOI", ooi_list)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"[OK] FSM_EdibleClass wrote: {OUT_PATH}")
    print("[DEBUG] Done.")

if __name__ == "__main__":
    main()
