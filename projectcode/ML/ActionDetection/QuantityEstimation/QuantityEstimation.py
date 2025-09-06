"""
QuantityEstimation.py
---------------------

Runner script that uses detectron_cupflow_config.json to:
1) Run Detectron2 to detect spoon-in-cup triggers and draw bounding boxes/ROIs
2) Generate ROI heatmaps and flow arrow overlays

Outputs are written to folders defined below via INPUT_DIR/OUTPUT_DIR overrides
or taken from the JSON config when not provided here.
"""

import os
import argparse
from pathlib import Path
import json
from projectcode.ML.ActionDetection.QuantityEstimation.DetectronCupFlowPipeline import DetectronCupFlowPipeline
from projectcode.ML.ActionDetection.QuantityEstimation.InflowStats import InflowStats

# -------- Set these to override config.json --------
# Leave as None to use values from the JSON config file
#INPUT_DIR: str | None = None  # e.g., "/app/mediaFiles/videos/AnnotateVideos/RawVideos/CoffeePour/CoffeePour"
#OUTPUT_DIR: str | None = None # e.g., "/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output/detectron2_cupflow_pipeline"

INPUT_DIR = '/app/mediaFiles/videos/AnnotateVideos/RawVideos/CoffeePour/CoffeePour'
OUTPUT_DIR = '/app/mediaFiles/output/videoOutputs/DetectedFrames/coffee_volume_output'

DEFAULT_CONFIG_PATH = "/app/projectcode/ML/ActionDetection/QuantityEstimation/detectron_cupflow_config.json"

def main():
    parser = argparse.ArgumentParser(description="Coffee quantity estimation pipeline (Detectron2 + ROI heat/flow)")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to JSON config (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    # Load config and inject INPUT_DIR/OUTPUT_DIR overrides if provided
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    if INPUT_DIR is not None:
        cfg["input_dir"] = INPUT_DIR
    if OUTPUT_DIR is not None:
        cfg["output_dir"] = OUTPUT_DIR

    # Ensure we now have both required paths
    if "input_dir" not in cfg or "output_dir" not in cfg or not cfg["input_dir"] or not cfg["output_dir"]:
        raise ValueError("Please set INPUT_DIR and OUTPUT_DIR at the top of QuantityEstimation.py or in the config JSON.")

    # Persist a transient runtime config next to the original so the pipeline can load it
    runtime_cfg_path = str(Path(cfg_path).with_name(Path(cfg_path).stem + ".runtime.json"))
    with open(runtime_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    pipe = DetectronCupFlowPipeline(config_path=runtime_cfg_path)

    # Phase 1+2: detect triggers, draw cup boxes and flow ROI, save frames + CSV + masks
    pipe.run_detection_and_rois()

    # Phase 3: generate ROI heatmaps and flow arrow overlays
    pipe.generate_heat_and_arrows()

    # Phase 4+5: filter rim noise and estimate quantity from heat scores
    inflow = InflowStats(config_path=runtime_cfg_path)
    inflow.run()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()


