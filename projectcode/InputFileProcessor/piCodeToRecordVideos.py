## this is pi code and rclone must be configured with Google Drive remote named 'gdrive' ## 
import sounddevice as sd
import queue
import vosk
import json
import subprocess
import os
import time
from datetime import datetime, timedelta
import glob

q = queue.Queue()
model = vosk.Model("/home/pi/vosk-model/model")
rec = vosk.KaldiRecognizer(model, 16000)

# Folder where local videos are stored
LOCAL_DIR = "/home/pi/Videos/"
os.makedirs(LOCAL_DIR, exist_ok=True)

def callback(indata, frames, time_, status):
    q.put(bytes(indata))

def cleanup_old_files():
    """Delete local files older than 3 days."""
    now = time.time()
    cutoff = now - (3 * 86400)  # 3 days in seconds

    for file in glob.glob(os.path.join(LOCAL_DIR, "*.mp4")):
        if os.path.getmtime(file) < cutoff:
            print(f"🧹 Deleting old file: {file}")
            os.remove(file)

def record_and_upload(duration_ms=120000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file = os.path.join(LOCAL_DIR, f"video_{timestamp}.h264")
    mp4_file = os.path.join(LOCAL_DIR, f"video_{timestamp}.mp4")

    print("🎥 Recording...")
    subprocess.run(["rpicam-vid", "-t", str(duration_ms), "-o", raw_file])

    print("⚙️ Converting to MP4...")
    subprocess.run([
        "ffmpeg", "-y", "-r", "30", "-i", raw_file,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_file
    ])

    # Remove raw .h264 after conversion
    if os.path.exists(raw_file):
        os.remove(raw_file)

    print("☁️ Uploading to Google Drive...")
    subprocess.run(["rclone", "copy", mp4_file, "gdrive:/RaspberryPiVideos/"])
    print(f"✅ Upload complete! File: {mp4_file}")

    # Cleanup old files
    cleanup_old_files()

def main():
    print("🔊 Say 'start recording' to capture a 2-minute video and upload to Drive.")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("Heard:", text)

                if "hey mace" in text.lower():
                    # Default duration 2 minutes
                    duration_ms = 120000

                    if "cooking" in text.lower():
                        duration_ms = 120000   # 2 minutes
                    elif "eating" in text.lower():
                        duration_ms = 20000   # 2 minutes + 20 seconds

                    record_and_upload(duration_ms=duration_ms)
                    print("👌 Done recording. Waiting for next command...")

if __name__ == "__main__":
    main()
