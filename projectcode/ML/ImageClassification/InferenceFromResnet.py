import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

# --- Config ---
CHECKPOINT_PATH = "/app/projectcode/ModelCheckpointsAndLabels/resnet50_food101_plus_newclasses_final.pth"
LABELS_PATH = "/app/projectcode/ModelCheckpointsAndLabels/resnet_101_labels.txt"
IMAGE_PATH = "/app/mediaFiles/images/ResnetTrainingImages/black coffee/20211008_090559.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load all labels (101 Food-101 + 3 new classes) ---
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Check total classes
assert len(labels) == 104, "Labels file must contain 104 entries!"

# Indices of new classes (last 3)
new_class_indices = [101, 102, 103]  # coffee, black coffee, banana

# --- Model loader ---
def get_model(num_classes=104):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

model = get_model()
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# --- Forward pass ---
with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)

# --- Print probabilities for the new classes only ---
print("\nProbabilities for coffee / black coffee / banana:")
for idx in new_class_indices:
    print(f"{labels[idx]}: {probs[0, idx].item():.6f}")

# --- Optional: top-3 predicted class among new classes ---
top3_probs, top3_indices = torch.topk(probs[0, new_class_indices], k=3)
print("\nTop prediction among new classes:")
for i in range(3):
    class_idx = new_class_indices[top3_indices[i].item()]
    print(f"{labels[class_idx]}: {top3_probs[i].item():.6f}")
