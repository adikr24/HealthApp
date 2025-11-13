import os
from pathlib import Path
import shutil

SRC_DIR = Path("/app/mediaFiles/videos/InputVideos/VideosToAnnotate/VideoInfFile/FishAndPotatoes/Step9_PouringOilAndSpicesOnSalmonAndTilapia")
DST_DIR = Path("/app/mediaFiles/videos/InputVideos/VideosToAnnotate/VideoInfFile/AnnotateImages_FishAndPotatoes/Step9_PouringOilAndSpicesOnSalmonAndTilapia")

# create destination folder
DST_DIR.mkdir(parents=True, exist_ok=True)

# get all jpg frames sorted
frames = sorted([p for p in SRC_DIR.glob("*.jpg")])

print(f"Found {len(frames)} frames in source folder.")

count = 0
for i, frame_path in enumerate(frames):
    if i % 5 == 0:
        dst = DST_DIR / frame_path.name
        shutil.copy(frame_path, dst)
        count += 1

print(f"Copied {count} sampled frames to {DST_DIR}")
