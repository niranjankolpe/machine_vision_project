"""
train.py
---------
Train the Cotton Disease Detection model using EfficientNet-B0 transfer learning.

Usage:
    python train.py
    python train.py --epochs 50 --batch_size 16 --lr 0.001
    python train.py --dataset_dir my_dataset --resume checkpoints/last.pth
"""

import os
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Transforms ─────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ── Dataset Loader ─────────────────────────────────────────────────────────────
def get_dataloaders(dataset_dir: Path, batch_size: int):
    train_ds = datasets.ImageFolder(dataset_dir / "train", transform=train_transforms)
    val_ds   = datasets.ImageFolder(dataset_dir / "val",   transform=val_transforms)
    test_ds  = datasets.ImageFolder(dataset_dir / "test",  transform=val_transforms)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    print(f"\n📦 Dataset loaded from: {dataset_dir}")
    print(f"   Classes  : {train_ds.classes}")
    print(f"   Train    : {len(train_ds)} images")
    print(f"   Val      : {len(val_ds)} images")
    print(f"   Test     : {len(test_ds)} images\n")

    return train_loader, val_loader, test_loader, train_ds.classes


# ── Training Loop ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Cotton Disease Classifier")
    parser.add_argument("--dataset_dir", default="dataset",   type=str)
    parser.add_argument("--epochs",      default=40,          type=int)
    parser.add_argument("--batch_size",  default=8,           type=int)
    parser.add_argument("--lr",          default=5e-4,        type=float)
    parser.add_argument("--weight_decay",default=1e-4,        type=float)
    parser.add_argument("--patience",    default=10,          type=int,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--resume",      default=None,        type=str,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🔧 Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_path = ROOT / args.dataset_dir
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_path, args.batch_size
    )
    num_classes = len(classes)

    # Save class order for inference
    with open(CHECKPOINT_DIR / "classes.json", "w") as f:
        json.dump(classes, f)

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.model import CottonDiseaseModel
    model = CottonDiseaseModel(num_classes=num_classes).to(device)

    # Class-balanced loss for small datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    start_epoch = 1
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Resume from checkpoint
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0)
        history = ckpt.get("history", history)
        print(f"▶ Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.2f}%)")

    print(f"\n🚀 Training for {args.epochs} epochs | Patience: {args.patience}\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("─" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            flag = " ✅ best"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": classes,
                "history": history,
            }, CHECKPOINT_DIR / "best.pth")
        else:
            patience_counter += 1

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "classes": classes,
            "history": history,
        }, CHECKPOINT_DIR / "last.pth")

        print(
            f"{epoch:>6}/{args.epochs}  "
            f"{train_loss:>10.4f}  {train_acc:>9.2f}%  "
            f"{val_loss:>9.4f}  {val_acc:>8.2f}%  "
            f"{elapsed:>5.1f}s{flag}"
        )

        if patience_counter >= args.patience:
            print(f"\n⏹  Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    print(f"\n📊 Loading best model for final test evaluation...")
    best_ckpt = torch.load(CHECKPOINT_DIR / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, preds, labels_true = evaluate(model, test_loader, criterion, device)

    print(f"\n{'─'*40}")
    print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Test Accuracy     : {test_acc:.2f}%")
    print(f"  Test Loss         : {test_loss:.4f}")
    print(f"{'─'*40}")

    # Per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)
    for p, t in zip(preds, labels_true):
        class_total[classes[t]] += 1
        if p == t:
            class_correct[classes[t]] += 1

    print("\n📋 Per-class Test Accuracy:")
    for cls in classes:
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
            print(f"   {cls:30s}: {acc:.1f}% ({class_correct[cls]}/{class_total[cls]})")

    print(f"\n✅ Training complete! Best model saved to: {CHECKPOINT_DIR}/best.pth")


if __name__ == "__main__":
    main()
