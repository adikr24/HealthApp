import sys
import os
import subprocess

# Define the base path to your SlowFast library
# MAKE SURE THIS PATH IS CORRECT FOR YOUR SETUP
SLOWFAST_BASE_PATH = r"C:\Users\adity\OneDrive\Desktop\Aditya\StartupIdeas\HealthMonitor\dependencies\MLlibs\SlowFast"

# Path to the pre-trained model checkpoint (already downloaded)
CHECKPOINT_FILE = r"C:\Users\adity\OneDrive\Desktop\Aditya\StartupIdeas\HealthMonitor\Model_Checkpoints\SlowFastTrainedModel\SLOWFAST_8x8_R50.pkl"

# Path to the input video you want to process
INPUT_VIDEO_PATH = r"C:\Users\adity\OneDrive\Desktop\Aditya\StartupIdeas\HealthMonitor\Data_Video_streams\pouringCoffee_compressed.mp4"

# Path for the output video (where classification results will be overlaid)
OUTPUT_VIDEO_PATH = r"C:\Users\adity\OneDrive\Desktop\Aditya\StartupIdeas\HealthMonitor\Data_Video_streams\pouringCoffee_output.mp4"

# Path to the label file for class names (usually found within SlowFast repo)
LABEL_FILE_PATH = os.path.join(SLOWFAST_BASE_PATH, r"label_files\kinetics_classnames.json")

# Path to the configuration file for the SlowFast_8x8_R50 model
CONFIG_FILE = os.path.join(SLOWFAST_BASE_PATH, r"configs\Kinetics\SlowFast_8x8_R50.yaml")

# A dummy data directory is often needed by the config, even if not actively used for inference
# You can point it to your video streams folder for simplicity
DATA_DIR = r"C:\Users\adity\OneDrive\Desktop\Aditya\StartupIdeas\HealthMonitor\Data_Video_streams"


# --- Construct the command to run tools/run_net.py ---
command = [
    sys.executable,  # Use the current Python interpreter
    os.path.join(SLOWFAST_BASE_PATH, "tools", "run_net.py"),
    "--cfg", CONFIG_FILE,
    "DATA.PATH_TO_DATA_DIR", DATA_DIR,
    "TRAIN.ENABLE", "False",  # Disable training
    "TEST.ENABLE", "True",    # Enable testing (inference)
    "TEST.CHECKPOINT_FILE_PATH", CHECKPOINT_FILE,
    "NUM_GPUS", "1",          # Use 1 GPU (your RTX 5060)
    "TEST.BATCH_SIZE", "1",   # Small batch size for single video inference
    "DEMO.ENABLE", "True",    # Enable the demo mode for processing a video
    "DEMO.INPUT_VIDEO", INPUT_VIDEO_PATH,
    "DEMO.OUTPUT_FILE", OUTPUT_VIDEO_PATH,
    "DEMO.LABEL_FILE_PATH", LABEL_FILE_PATH
]

print("--- Starting SlowFast Inference Test Run ---")
print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Output will be saved to: {OUTPUT_VIDEO_PATH}")
print(f"Using Checkpoint: {CHECKPOINT_FILE}")
print(f"Executing command: {' '.join(command)}\n")

try:
    # Create a copy of the current environment variables
    env = os.environ.copy()
    
    # Add SLOWFAST_BASE_PATH to PYTHONPATH for the subprocess
    # This tells the subprocess where to find the 'slowfast' package
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{SLOWFAST_BASE_PATH}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = SLOWFAST_BASE_PATH

    # Run the command using subprocess, capturing stdout and stderr for debugging
    # `cwd` ensures the script runs from the SlowFast base directory,
    # which helps with any relative paths or internal imports.
    result = subprocess.run(
        command,
        check=True,
        cwd=SLOWFAST_BASE_PATH,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("\n--- SlowFast Inference Test Run Completed Successfully! ---")
    print(f"Check '{OUTPUT_VIDEO_PATH}' for the annotated video.")
    print("\n--- Subprocess Output ---\n")
    print(result.stdout)
    if result.stderr:
        print("\n--- Subprocess Error Output ---\n")
        print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"\n--- An error occurred during SlowFast Inference Test Run: {e} ---")
    print("Please ensure all paths are correct and the checkpoint.py patch is applied.")
    print("If an error persists, please share the full traceback.")
    if hasattr(e, 'stdout') and e.stdout:
        print("\n--- Subprocess Output ---\n")
        print(e.stdout)
    if hasattr(e, 'stderr') and e.stderr:
        print("\n--- Subprocess Error Output ---\n")
        print(e.stderr)
except FileNotFoundError:
    print(f"\n--- Error: The Python executable or run_net.py script was not found. ---")
    print(f"Please ensure '{sys.executable}' exists and '{os.path.join(SLOWFAST_BASE_PATH, 'tools', 'run_net.py')}' is the correct path.")
except Exception as e:
    print(f"\n--- An unexpected error occurred: {e} ---")
    import traceback
    traceback.print_exc()
