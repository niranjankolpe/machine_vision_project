"""
train_mangonet.py
=================
Trains MangoNet CNN on mango dataset.
Input : dataset/harvest/ and dataset/raw/ images
Output: models/mangonet.pth

HOW TO RUN:
  python train_mangonet.py

Prints progress every epoch. Takes ~35-40 mins on CPU for 50 epochs.
"""

import os
import glob
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODELS_DIR   = "models"
EPOCHS       = 50
BATCH_SIZE   = 4
LR           = 1e-3
AUG_FACTOR   = 8      # each image repeated 8x with random augmentation
SEED         = 42

os.makedirs(MODELS_DIR, exist_ok=True)
torch.manual_seed(SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Harvest Ready", "Raw (Not Ready)"]

# ── Transforms ────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])


# ── Dataset ───────────────────────────────────────────────────────────
class MangoDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img    = Image.open(self.paths[idx]).convert('RGB')
        tensor = TRAIN_TRANSFORM(img) if self.augment else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])(img)
        return tensor, self.labels[idx]


# ── MangoNet Architecture ─────────────────────────────────────────────
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class MangoNet(nn.Module):
    """
    Custom lightweight CNN for mango classification.

    4 convolutional blocks:
      Block 1: detects edges and basic color patterns
      Block 2: detects texture patterns (lenticels, surface)
      Block 3: detects shape and regional features
      Block 4: high-level mango representations

    AdaptiveAvgPool -> 256-dim vector -> Dropout -> Linear(2)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    nn.MaxPool2d(2, 2),   # 112x112
            conv_block(32, 64),   nn.MaxPool2d(2, 2),   # 56x56
            conv_block(64, 128),  nn.MaxPool2d(2, 2),   # 28x28
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))                # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load Dataset ──────────────────────────────────────────────────────
def load_dataset():
    harvest = sorted(glob.glob(f"{DATASET_DIR}/harvest/*"))
    raw     = sorted(glob.glob(f"{DATASET_DIR}/raw/*"))
    paths   = harvest + raw
    labels  = [0] * len(harvest) + [1] * len(raw)

    print(f"  Harvest-ready : {len(harvest)} images")
    print(f"  Raw           : {len(raw)} images")
    print(f"  Total         : {len(paths)} images")
    print(f"  After aug x{AUG_FACTOR}: {len(paths) * AUG_FACTOR} samples")
    return paths, labels


# ── Training Loop ─────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  MANGONET TRAINING")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}\n")

    print("  Loading dataset...")
    paths, labels = load_dataset()

    # Augment by repeating paths
    aug_paths  = paths  * AUG_FACTOR
    aug_labels = labels * AUG_FACTOR

    dataset = MangoDataset(aug_paths, aug_labels, augment=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = MangoNet(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  MangoNet parameters: {total_params:,}")
    print(f"\n  {'Epoch':<8} {'Loss':<10} {'Acc':<10} {'Time':<10} ETA")
    print(f"  {'-'*52}")

    loss_history = []
    train_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        correct = 0
        total   = 0

        for tensors, batch_labels in loader:
            tensors      = tensors.to(DEVICE)
            batch_labels = torch.as_tensor(batch_labels).clone().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(tensors)
            loss    = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            preds    = torch.argmax(outputs, dim=1)
            correct += (preds == batch_labels).sum().item()
            total   += len(batch_labels)

        scheduler.step()

        avg_loss = ep_loss / len(loader)
        acc      = correct / total * 100
        loss_history.append(avg_loss)

        elapsed          = time.time() - train_start
        time_per_epoch   = elapsed / epoch
        remaining        = time_per_epoch * (EPOCHS - epoch)
        eta_m, eta_s     = int(remaining) // 60, int(remaining) % 60
        el_m,  el_s      = int(elapsed)   // 60, int(elapsed)   % 60

        # Progress bar
        done = int((epoch / EPOCHS) * 20)
        bar  = "[" + "#" * done + "-" * (20 - done) + "]"

        print(
            f"  {epoch:3d}/{EPOCHS}  {bar}  "
            f"loss={avg_loss:.4f}  acc={acc:.1f}%  "
            f"{el_m:02d}m{el_s:02d}s  ETA {eta_m:02d}m{eta_s:02d}s",
            flush=True
        )

    # Save model
    save_path = os.path.join(MODELS_DIR, "mangonet.pth")
    torch.save(model.state_dict(), save_path)

    total_time = time.time() - train_start
    print(f"\n  Training complete in "
          f"{int(total_time)//60}m {int(total_time)%60}s")
    print(f"  Model saved -> {save_path}")

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color='steelblue', linewidth=2)
    plt.title("MangoNet Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = os.path.join(MODELS_DIR, "loss_curve.png")
    plt.savefig(curve_path, dpi=120)
    plt.close()
    print(f"  Loss curve  -> {curve_path}")

    print("\n" + "=" * 60)
    print("  DONE. models/mangonet.pth is ready.")
    print("  Run python gui.py to use the model.")
    print("=" * 60)


if __name__ == "__main__":
    train()
