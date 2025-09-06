import torch
import torchvision.transforms as transforms
from src.models.tsm import TSM
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# -------------------------------
# Load TSM model
# -------------------------------
def load_tsm_model(checkpoint_path, num_class=397, num_segments=8, modality='RGB', device='cpu'):
    model = TSM(num_class=num_class, num_segments=num_segments, modality=modality)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model

# -------------------------------
# Load action classes
# -------------------------------
def load_action_classes(csv_path):
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].apply(lambda x: int(eval(x)) if isinstance(x, str) and x.startswith("[") else int(x))
    return dict(zip(df["id"], df["action"]))

# -------------------------------
# Sample frames from video
# -------------------------------
def sample_frames(video_path, fps_sample=8, size=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps / fps_sample), 1)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    frames = []
    frame_indices = []

    for idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frame = preprocess(frame)
        frames.append(frame)
        frame_indices.append(idx)

    cap.release()
    return frames, frame_indices, video_fps

# -------------------------------
# Chunk frames into segments for TSM
# -------------------------------
def chunk_frames(frames, chunk_size=8):
    chunks = []
    chunk_starts = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk += [chunk[-1]] * (chunk_size - len(chunk))  # pad last frame
        chunks.append(torch.stack(chunk).unsqueeze(0))  # [1, chunk_size, 3, H, W]
        chunk_starts.append(i)
    return chunks, chunk_starts

# -------------------------------
# Run full inference
# -------------------------------
def run_inference(video_path, checkpoint_path, action_csv, device='cpu', fps_sample=8):
    action_map = load_action_classes(action_csv)
    num_classes = len(action_map)

    model = load_tsm_model(checkpoint_path, num_class=num_classes, device=device)
    frames, frame_indices, video_fps = sample_frames(video_path, fps_sample=fps_sample)
    
    type(sec_probs)
    len(sec_probs)
    sec_probs.shape
    
    chunks, chunk_starts = chunk_frames(frames, chunk_size=8)

    # Store chunk probabilities
    per_chunk_probs = []
    for chunk in chunks:
        chunk = chunk.to(device)
        with torch.no_grad():
            outputs = model(chunk)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            if probs.dim() == 2 and probs.shape[0] > 1:
                probs = probs.mean(dim=0)
            per_chunk_probs.append(probs.cpu())

    # -------------------------------
    # Per-second top-5 predictions
    # -------------------------------
    sec_probs = defaultdict(list)
    for prob, start_idx in zip(per_chunk_probs, chunk_starts):
        second = int(start_idx / video_fps)
        sec_probs[second].append(prob)

    sec_top5 = {}
    for sec, probs_list in sec_probs.items():
        avg_prob = torch.stack(probs_list).mean(dim=0).squeeze()  # remove extra dimensions
        k = min(5, avg_prob.numel())
        top5_idx = torch.topk(avg_prob, k).indices.tolist()  # now this will be a list of ints
        sec_top5[sec] = [(action_map[i], avg_prob[i].item()) for i in top5_idx]

    # -------------------------------
    # Overall top-5 prediction
    # -------------------------------
    all_probs = torch.stack(per_chunk_probs).mean(dim=0).squeeze()  # remove extra dimensions
    k = min(5, all_probs.numel())
    overall_top5_idx = torch.topk(all_probs, k).indices.tolist()  # now a flat list of ints
    overall_top5 = [(action_map[i], all_probs[i].item()) for i in overall_top5_idx]

    # -------------------------------
    # Print results
    # -------------------------------
    print("\n⏱ Per-second top-5 predictions:")
    for sec in sorted(sec_top5):
        print(f"{sec:02d}s → {sec_top5[sec]}")

    print(f"\n🎯 Overall top-5 predictions for video: {overall_top5}")
    return sec_top5, overall_top5

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video_path = "/app/mediaFiles/videos/labeledvideos/P03_25.MP4"
    checkpoint_path = "/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/models/tsm_rgb.ckpt"
    action_csv = "/app/mediaFiles/videos/kinetic_400_label/tsm_397_actions.csv"

    sec_top5, overall_top5 = run_inference(video_path, checkpoint_path, action_csv, device=device)
