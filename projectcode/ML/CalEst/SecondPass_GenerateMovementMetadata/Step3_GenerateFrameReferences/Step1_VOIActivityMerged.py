#!/usr/bin/env python3
"""
Merge VOI activity rows by simple temporal overlap, using a precomputed labels CSV
(with column 'filterVOINoise') to ignore noise classes (e.g., lid).

Inputs (hard-coded to your paths as requested):
- Activity CSV:
  /app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/voi_activity_summary.csv
- Labels CSV (already includes filterVOINoise column you prepared):
  /app/mediaFiles/images/StaticCSVLists/YoloLabels.csv

Output:
- /app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/ActivityFrameSegmentation/all_activity_merged_frames.csv

Merging Rules:
- Sort by start_idx.
- Drop rows where class is in ignore set derived from labels[filterVOINoise=='Y'].
- Merge rows transitively if they overlap OR their gap <= MAX_GAP frames.
- A state's classes = union of member classes in FIRST-SEEN order (pipe-joined).
- start_frame = earliest start_frame in the merged window, end_frame = latest end_frame.

To adjust tolerance, change MAX_GAP below.
"""

import csv
from collections import OrderedDict
from pathlib import Path
import pandas as pd

# ----- CONFIG -----
ACTIVITY_CSV = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/voi_activity_summary.csv")
LABELS_CSV   = Path("/app/mediaFiles/images/StaticCSVLists/YoloLabels.csv")
OUT_CSV      = Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/SecondPass_GenerateMovementMetadata/Metadata/ActivityFrameSegmentation/all_activity_merged_frames.csv")

MAX_GAP = 10  # frames; set to 0 if you want strict overlap only

# ----- LOAD LABELS & BUILD IGNORE SET -----
labels = pd.read_csv(LABELS_CSV)
labels.columns = [c.strip() for c in labels.columns]
if "YoloAndDetectronTrainedClasses" not in labels.columns or "filterVOINoise" not in labels.columns:
    raise ValueError("Labels CSV must include 'YoloAndDetectronTrainedClasses' and 'filterVOINoise' columns.")
ignore_classes = set(
    labels.loc[labels["filterVOINoise"].astype(str).str.upper() == "Y", "YoloAndDetectronTrainedClasses"]
    .astype(str).str.strip().str.lower()
)

# ----- LOAD ACTIVITY -----
df = pd.read_csv(ACTIVITY_CSV)
df.columns = [c.strip().lower() for c in df.columns]

required = {"class","start_frame","end_frame","start_idx","end_idx","avg_dist","min_dist","overlap_hits","VOI"}
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Activity CSV missing required columns: {missing}")

# Normalize and filter
df["class"] = df["class"].astype(str).str.strip().str.lower()
df = df.sort_values(["start_idx","end_idx"], kind="mergesort").reset_index(drop=True)
df = df[~df["class"].isin(ignore_classes)].reset_index(drop=True)

# ----- MERGE BY OVERLAP (WITH SMALL GAP) -----
merged_states = []
if not df.empty:
    cur_start_idx = int(df.loc[0, "start_idx"])
    cur_end_idx   = int(df.loc[0, "end_idx"])
    cur_start_frame = str(df.loc[0, "start_frame"])
    cur_end_frame   = str(df.loc[0, "end_frame"])
    cur_voi = str(df.loc[0, "voi"])
    seen_classes = OrderedDict()
    seen_classes[str(df.loc[0, "class"])] = True

    for i in range(1, len(df)):
        row = df.loc[i]
        s, e = int(row["start_idx"]), int(row["end_idx"])
        # If this segment starts before or at (current end + MAX_GAP), merge it
        if s <= cur_end_idx + MAX_GAP:
            # extend time window
            if e > cur_end_idx:
                cur_end_idx = e
                cur_end_frame = str(row["end_frame"])
            # add class in first-seen order
            c = str(row["class"])
            if c not in seen_classes:
                seen_classes[c] = True
        else:
            # flush current state
            classes_str = "|".join(seen_classes.keys())
            merged_states.append({
                "start_idx": cur_start_idx,
                "end_idx": cur_end_idx,
                "start_frame": cur_start_frame,
                "end_frame": cur_end_frame,
                "classes": classes_str,
                "VOI": cur_voi
            })
            # start a new state
            cur_start_idx = s
            cur_end_idx   = e
            cur_start_frame = str(row["start_frame"])
            cur_end_frame   = str(row["end_frame"])
            seen_classes = OrderedDict()
            seen_classes[str(row["class"])] = True

    # flush last
    classes_str = "|".join(seen_classes.keys())
    merged_states.append({
        "start_idx": cur_start_idx,
        "end_idx": cur_end_idx,
        "start_frame": cur_start_frame,
        "end_frame": cur_end_frame,
        "classes": classes_str,
        "VOI": cur_voi
    })

# ----- BUILD OUTPUT DF -----
out_df = pd.DataFrame([{
    "state_id": i+1,
    "start_idx": st["start_idx"],
    "end_idx": st["end_idx"],
    "frames": int(st["end_idx"]) - int(st["start_idx"]) + 1,
    "start_frame": st["start_frame"],
    "end_frame": st["end_frame"],
    "classes": st["classes"],
    "VOI": st["VOI"]
} for i, st in enumerate(merged_states)])

# Ensure destination directory exists
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_CSV, index=False)

print(f"Saved merged states ({len(out_df)} rows) -> {OUT_CSV}")
