import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
import glob, os
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load pretrained Food-101 ResNet50
# -------------------------
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 101)  # temporarily 101


# Load pretrained Food-101 weights
model = model.to(device) 
state_dict_url = "https://huggingface.co/nateraw/food/resolve/main/pytorch_model.bin"
state_dict = torch.hub.load_state_dict_from_url(state_dict_url, map_location=device)

model.load_state_dict(state_dict)

# -------------------------
# Expand FC to 104 classes properly
# -------------------------
num_new_classes = 101 + 3
new_fc = nn.Linear(num_ftrs, num_new_classes)
with torch.no_grad():
    new_fc.weight[:101] = model.fc.weight.clone()  # clone to preserve pretrained weights
    new_fc.bias[:101] = model.fc.bias.clone()
model.fc = new_fc.to(device)

# -------------------------
# Freeze early layers, unfreeze last blocks
# -------------------------
for name, param in model.named_parameters():
    param.requires_grad = False
for param in model.layer3.parameters():  # unfreeze layer3 + layer4 + fc
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# -------------------------
# Data transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------
# Custom Dataset
# -------------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, label, transform=None):
        self.image_paths = image_paths
        self.label = label
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label

# -------------------------
# Load your 3 new-class datasets
# -------------------------
coffee_images = glob.glob("/app/mediaFiles/images/ResnetTrainingImages/coffee/*.jpg")
blackcoffee_images = glob.glob("/app/mediaFiles/images/ResnetTrainingImages/black coffee/*.jpg")
banana_images = glob.glob("/app/mediaFiles/images/ResnetTrainingImages/banana/*.jpg")

coffee_dataset = CustomImageDataset(coffee_images, 101, transform)
blackcoffee_dataset = CustomImageDataset(blackcoffee_images, 102, transform)
banana_dataset = CustomImageDataset(banana_images, 103, transform)

combined_dataset = ConcatDataset([coffee_dataset, blackcoffee_dataset, banana_dataset])

# Train/validation split
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------
# Loss & optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# -------------------------
# Training loop
# -------------------------
epochs = 5
best_val_loss = float('inf')
patience = 3
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    train_loss = running_loss / total
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()
    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | "
          f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
    
    scheduler.step()

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "resnet50_best_val_loss.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs")
            break

# -------------------------
# Save final model
# -------------------------
model.load_state_dict(torch.load("resnet50_best_val_loss.pth"))
torch.save(model.state_dict(), "resnet50_food101_plus_newclasses_final.pth")
print("✅ Training complete. Best model saved.")
