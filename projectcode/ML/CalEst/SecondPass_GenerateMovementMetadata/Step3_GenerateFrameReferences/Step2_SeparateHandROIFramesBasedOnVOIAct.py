#!/usr/bin/env python3
"""
Step 2 — Gate to FSMs for Hand-Only, Hand+Bottle, and Edible classes.

- Reads the activity segmentation CSV.
- Checks:
  • 'hand'-only  → FSM_HandOnlyClass
  • 'hand+bottle' → FSM_HandBottleClass
  • edible class (based on YoloLabels.csv) → FSM_EdibleClass
- Each FSM overwrites its own fsm_heuristics.csv with 'status' inserted after 'classes'.
"""

from pathlib import Path
import pandas as pd
import sys

# ---------- Import FSMs ----------
from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.Step3_GenerateFrameReferences.FSM_Heurisitc_HandOnlyClass.FSM_HandOnlyClass import FSM as HandOnlyFSM
from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.Step3_GenerateFrameReferences.FSM_Heuristic_HandBottleClass.FSM_HandBottleClass import FSM as HandBottleFSM
from projectcode.ML.CalEst.SecondPass_GenerateMovementMetadata.Step3_GenerateFrameReferences.FSM_Heuristic_EdibleClass.FSM_EdibleClass import FSM as EdibleFSM
# ---------- Base paths ----------
BASE = Path("/app/mediaFiles/output/videoOutputs/ProteinShakeIII/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata")
ACT_MERGED = BASE / "ActivityFrameSegmentation" / "all_activity_merged_frames.csv"
ACT_ALT    = BASE / "ActivityFrameSegmentation" / "all_activity_frames.csv"
YOLO_LABELS = Path("/app/mediaFiles/StaticCSVLists/YoloLabels.csv")

# ---------- Helpers ----------
def _load_activity_csv() -> pd.DataFrame:
    """Load whichever activity CSV exists."""
    if ACT_MERGED.exists():
        df = pd.read_csv(ACT_MERGED)
    elif ACT_ALT.exists():
        df = pd.read_csv(ACT_ALT)
    else:
        raise FileNotFoundError(f"Could not find activity CSV at {ACT_MERGED} or {ACT_ALT}")
    df.columns = [c.strip() for c in df.columns]
    return df

def _has_hand_only(df: pd.DataFrame) -> bool:
    return df["classes"].astype(str).str.strip().str.lower().eq("hand").any()

def _has_hand_bottle(df: pd.DataFrame) -> bool:
    def _has_both(s: str) -> bool:
        toks = {t.strip().lower() for t in str(s).split("|") if t.strip()}
        return "hand" in toks and "bottle" in toks
    return df["classes"].astype(str).map(_has_both).any()

def _has_edible(df: pd.DataFrame) -> bool:
    """Check if any activity row contains an edible class using YoloLabels.csv."""
    labels = pd.read_csv(YOLO_LABELS)
    labels.columns = [c.strip() for c in labels.columns]
    col_class = next(c for c in labels.columns if c.lower().startswith("yoloanddetectrontrainedclasses"))
    col_edible = next(c for c in labels.columns if c.lower() == "edible")

    edible_set = {
        str(cls).strip().lower()
        for cls, edible_flag in zip(labels[col_class], labels[col_edible])
        if str(edible_flag).strip().upper() == "Y"
    }

    for _, row in df.iterrows():
        tokens = [t.strip().lower() for t in str(row["classes"]).split("|") if t.strip()]
        if any(t in edible_set for t in tokens):
            return True
    return False

# ---------- Main ----------
def main() -> int:
    try:
        act = _load_activity_csv()
    except Exception as e:
        print(f"[ERROR] Step2: {e}", file=sys.stderr)
        return 2

    ran_fsm = False

    # --- Hand-only FSM ---
    try:
        if _has_hand_only(act):
            HandOnlyFSM().run()
            print("[OK] Step2: Hand-only FSM executed and fsm_heuristics.csv written.")
            ran_fsm = True
    except Exception as e:
        print(f"[ERROR] Step2: Hand-only FSM failed: {e}", file=sys.stderr)
        return 3

    # # --- Hand+bottle FSM ---
    # try:
    #     if _has_hand_bottle(act):
    #         HandBottleFSM().run()
    #         print("[OK] Step2: Hand+bottle FSM executed and fsm_heuristics.csv written.")
    #         ran_fsm = True
    # except Exception as e:
    #     print(f"[ERROR] Step2: Hand+bottle FSM failed: {e}", file=sys.stderr)
    #     return 4

    # --- Edible FSM ---
    try:
        if _has_edible(act):
            EdibleFSM().run()
            print("[OK] Step2: Edible FSM executed and fsm_heuristics.csv written.")
            ran_fsm = True
    except Exception as e:
        print(f"[ERROR] Step2: Edible FSM failed: {e}", file=sys.stderr)
        return 5

    if not ran_fsm:
        print("[OK] Step2: No relevant (hand, hand+bottle, or edible) segments found — skipping FSMs.")
    return 0

if __name__ == "__main__":
    main()
