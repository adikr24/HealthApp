import torch
import torch.nn.functional as F
from pytorchvideo.models.hub import x3d_s
from pytorchvideo.data.encoded_video import EncodedVideo
import csv
import math

# -------------------------------
# 1. Load Kinetics-400 labels
# -------------------------------
kinetics_labels = {}
with open("/app/mediaFiles/videos/kinetic_400_label/kinetics_400_labels.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        idx, label = row
        kinetics_labels[str(idx)] = label

# -------------------------------
# 2. Device and model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = x3d_s(pretrained=True).eval().to(device)

# -------------------------------
# 3. Preprocessing function
# -------------------------------
def preprocess_clip(clip, num_frames=16, spatial_size=256):
    T, H, W, C = clip.shape
    clip = clip[:, :, :, :3]  # Ensure 3 channels

    # Temporal subsample / pad
    if T >= num_frames:
        indices = torch.linspace(0, T-1, steps=num_frames).long()
        clip = clip[indices]
    else:
        pad = clip[-1:].repeat(num_frames - T, 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)

    # Convert to (T,C,H,W) and scale
    clip = clip.permute(0, 3, 1, 2).float() / 255.0
    clip = F.interpolate(clip, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False)

    # Normalize
    mean = torch.tensor([0.45, 0.45, 0.45], device=clip.device).view(1,3,1,1)
    std  = torch.tensor([0.225, 0.225, 0.225], device=clip.device).view(1,3,1,1)
    clip = (clip - mean) / std

    # Rearrange for model: (1,C,T,H,W)
    clip = clip.permute(1,0,2,3).unsqueeze(0)
    return clip

# -------------------------------
# 4. Sliding-window inference function
# -------------------------------
def run_inference(video_path, start_sec=0, end_sec=20.0, num_frames=16, step_frames=8):
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
    T, H, W, C = clip.shape

    print(f"\nClip: {video_path}")
    print(f"  dtype: {clip.dtype}")
    print(f"  shape (T,H,W,C): {clip.shape}")
    print(f"  min value: {clip.min().item():.4f}")
    print(f"  max value: {clip.max().item():.4f}")

    probs_list = []

    # Slide over the video
    for start in range(0, T, step_frames):
        end = min(start + num_frames, T)
        window = clip[start:end]
        window_tensor = preprocess_clip(window, num_frames=num_frames).to(device)

        with torch.no_grad():
            preds = model(window_tensor)
            probs = torch.nn.functional.softmax(preds, dim=1)
            probs_list.append(probs)

    # Aggregate probabilities
    all_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    top5 = torch.topk(all_probs, 5)
    return top5

# -------------------------------
# 5. Run on video files
# -------------------------------
video_files = ["/app/mediaFiles/videos/labeledvideos/pouringMilkandsugar_trimmed.mp4"]

for vf in video_files:
    top5 = run_inference(vf, start_sec=0, end_sec=20.0)
    print(f"\nTop 5 predictions for {vf}:")
    for score, idx in zip(top5.values[0], top5.indices[0]):
        label = kinetics_labels.get(str(idx.item()), "Unknown")
        print(f"  Class {idx.item()} ({label}): {score.item():.4f}")
