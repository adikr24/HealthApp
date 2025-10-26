#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# your two images
IMG_PATHS = [
    Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/band_debug/diff_heat/diff_01068_01069.png"),
    Path("/app/mediaFiles/output/videoOutputs/ProteinShake/CookingAnalysis/ThirdPass_DetectVolume/band_debug/diff_heat/diff_01046_01047.png"),
]

for path in IMG_PATHS:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERROR] cannot read {path}")
        continue

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    print(f"\nImage: {path.name}")
    print(f"Shape: (height={h}, width={w}, channels={c})")
    print(f"Dtype: {img_rgb.dtype}")
    print("First 10 pixels (RGB):")
    print(img_rgb.reshape(-1, 3)[:100])

img_bgr[10]
