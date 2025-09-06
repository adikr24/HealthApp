#sorry I meant this code - import torch
import requests
import json
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms

# -----------------------
# Custom Transform Functions
# -----------------------
class UniformTemporalSubsample(torch.nn.Module):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, frames: torch.Tensor):
        # frames: (C, T, H, W)
        t_indices = torch.linspace(0, frames.shape[1] - 1, self.num_samples).long()
        return torch.index_select(frames, 1, t_indices)

class ShortSideScale(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, frames: torch.Tensor):
        # frames: (C, T, H, W)
        C, T, H, W = frames.shape
        if H < W:
            new_h = self.size
            new_w = int(W / H * self.size)
        else:
            new_w = self.size
            new_h = int(H / W * self.size)
        frames_resized = torch.nn.functional.interpolate(
            frames, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        return frames_resized

class CenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, frames: torch.Tensor):
        # frames: (C, T, H, W)
        H, W = frames.shape[2], frames.shape[3]
        top = (H - self.size) // 2
        left = (W - self.size) // 2
        return frames[:, :, top:top + self.size, left:left + self.size]

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1, 1)

    def forward(self, frames: torch.Tensor):
        return (frames - self.mean) / self.std

class PackPathway(torch.nn.Module):
    def __init__(self, slowfast_alpha):
        super().__init__()
        self.slowfast_alpha = slowfast_alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha).long(),
        )
        return [slow_pathway, fast_pathway]

# -----------------------
# Helper: Get top prediction
# -----------------------
def get_top_prediction(preds):
    url = "https://huggingface.co/datasets/huggingface/label-files/raw/main/kinetics400-id2label.json"
    response = requests.get(url)
    response.raise_for_status()
    labels = json.loads(response.text)

    probs = torch.nn.functional.softmax(preds, dim=1)
    top_prob, top_idx = torch.topk(probs, 1)
    idx = top_idx[0].item()
    return labels.get(str(idx), "Unknown"), top_prob.item()

# -----------------------
# Settings
# -----------------------
video_path = "/app/mediaFiles/videos/inputvideos/pouringCoffee_compressed.mp4"
slowfast_alpha = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load video and preprocess
# -----------------------
video = EncodedVideo.from_path(video_path)
video_tensor = video.get_clip(0, video.duration)["video"]  # (C, T, H, W)

class Div255(torch.nn.Module):
    def forward(self, x):
        return x / 255.0
    
transform = torch.nn.Sequential(
    UniformTemporalSubsample(32),
    Div255(),
    ShortSideScale(256),
    CenterCrop(256),
    Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    PackPathway(slowfast_alpha)
)

inputs = transform(video_tensor)
print(f"Slow path shape: {inputs[0].shape}")
print(f"Fast path shape: {inputs[1].shape}")

# -----------------------
# Load SlowFast model
# -----------------------
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
model = model.eval().to(device)

# -----------------------
# Inference
# -----------------------
with torch.no_grad():
    preds = model([i.unsqueeze(0).to(device) for i in inputs])

label, conf = get_top_prediction(preds)

print("-" * 20)
print(f"Predicted Action: {label} (Confidence: {conf:.4f})")
print("-" * 20)
