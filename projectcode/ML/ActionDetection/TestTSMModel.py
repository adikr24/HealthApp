import os
import torch
import torchvision.transforms as transforms
from src.models.tsm import TSM
from torch.utils.data import DataLoader
#from dataset import VideoDataset  # you'll need a simple dataset class to load video frames
import cv2
import numpy as np
import pandas as pd


def load_tsm_model(checkpoint_path, num_class=125, num_segments=8, modality='RGB', device='cpu'):
    model = TSM(num_class=num_class, num_segments=num_segments, modality=modality)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    # remove "module." prefix if checkpoint was saved from DataParallel
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    model.eval()
    model.to(device)
    return model

def preprocess_frames(video_path, num_segments=8, size=224):
    """Extract num_segments frames from video evenly spaced and transform to tensor"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_indices = np.linspace(0, total_frames - 1, num_segments, dtype=int)
    
    frames = []
    for idx in segment_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size))
        frame = transforms.ToTensor()(frame)
        frames.append(frame)
    
    cap.release()
    frames = torch.stack(frames)  # Shape: (num_segments, 3, H, W)
    frames = frames.unsqueeze(0)   # Add batch dimension -> (1, num_segments, 3, H, W)
    return frames

def run_inference(video_path, checkpoint_path, device='cpu'):
    model = load_tsm_model(checkpoint_path, device=device)
    frames = preprocess_frames(video_path)
    frames = frames.to(device)
    
    with torch.no_grad():
        outputs = model(frames)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    
    print(f"Inference complete. Predicted class index: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    video_path = "/app/mediaFiles/videos/labeledvideos/P06_14.MP4"
    checkpoint_path = "/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/models/tsm_rgb.ckpt"
    run_inference(video_path, checkpoint_path, device='cpu')

#classes_file_path = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_verb_classes.csv"
