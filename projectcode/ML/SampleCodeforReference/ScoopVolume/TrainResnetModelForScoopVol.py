#!/usr/bin/env python3
# ------------------------------------------------------------
# Train a ResNet-18 classifier for fill-level detection
# (empty / half / full) for scoop/spoon images.
# All parameters are pre-configured — just run in VS Code.
# ------------------------------------------------------------

from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIGURATION ----------------
DATA_ROOT = Path("/app/mediaFiles/output/videoOutputs/ScoopAnalysis")
MODEL_OUT = Path("/app/mediaFiles/output/videoOutputs/ScoopAnalysis/models/filllevel_resnet18.pt")

IMG_SIZE     = 224
EPOCHS       = 15
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-2
VAL_SPLIT    = 0.15
NUM_WORKERS  = 0     # safer inside Docker / WSL2
# ------------------------------------------------


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def split_dataset(full_ds, val_split):
    n_total = len(full_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    return torch.utils.data.random_split(full_ds, [n_train, n_val])


def make_loaders():
    train_tf, eval_tf = build_transforms()
    full_ds = datasets.ImageFolder(str(DATA_ROOT), transform=train_tf)
    train_ds, val_ds = split_dataset(full_ds, VAL_SPLIT)
    val_ds.dataset.transform = eval_tf

    targets = [y for _, y in train_ds]
    counts = Counter(targets)
    weights = torch.DoubleTensor([1.0 / counts[y] for y in targets])
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, full_ds.classes


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
    return loss_sum / total, correct / total, all_true, all_pred


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = make_loaders()
    print("Detected classes:", class_names)

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:02d}] Train {tr_loss:.4f}/{tr_acc:.3f} | Val {val_loss:.4f}/{val_acc:.3f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, model.state_dict()

    if best_state:
        # auto-create output directory
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dict": best_state,
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }, MODEL_OUT)
        print(f"\n✅ Saved best model ({best_acc:.3f}) → {MODEL_OUT}")
        print("Classes:", class_names)

        model.load_state_dict(best_state)
        _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))
        print("\nReport:\n", classification_report(y_true, y_pred, target_names=class_names, digits=3))
    else:
        print("⚠️ Training finished, but no improvement recorded.")


if __name__ == "__main__":
    main()
