import subprocess

cmd = [
    "yolo",
    "detect",
    "train",
    "data=/app/mediaFiles/images/YoloTrainingImages/yolotrainingImages/data.yaml",
    "model=yolov8n.pt",
    "epochs=50",
    "imgsz=640",
    "workers=0",   # single-threaded to avoid shared memory issues
    "batch=4"      # optional: smaller batch to reduce memory usage
]

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

for line in process.stdout:
    print(line, end="")

process.wait()
print(f"YOLO training finished with exit code {process.returncode}")