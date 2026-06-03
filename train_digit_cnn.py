"""
Training script for the handwritten digit CNN.

Trains on EMNIST (digits split) then optionally fine-tunes on custom data.
Saves the model to models/digit_cnn.pth.

Usage:
    python train_digit_cnn.py [--epochs 10] [--custom-data path/to/custom]

The trained model is used by utils/ocr_reader.py for handwritten digit
recognition on exam answer sheets.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils.ocr_reader import _DigitCNN


# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
LR           = 1e-3
EPOCHS_EMNIST = 5
EPOCHS_FINETUNE = 3
MODEL_PATH   = Path("models/digit_cnn.pth")

# ── Data augmentation ──────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_EMNIST)
    parser.add_argument("--custom-data", type=str, default=None,
                        help="Path to custom digit dataset (ImageFolder format)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MODEL_PATH.parent.mkdir(exist_ok=True)

    model = _DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ── EMNIST pre-training ────────────────────────────────────────────────
    print("Loading EMNIST digits …")
    emnist_train = datasets.EMNIST(root="data", split="digits", train=True,
                                   download=True, transform=train_transform)
    emnist_val   = datasets.EMNIST(root="data", split="digits", train=False,
                                   download=True, transform=val_transform)
    train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2)
    val_loader   = DataLoader(emnist_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"  Epoch {epoch}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

    # ── Optional fine-tuning on custom data ───────────────────────────────
    if args.custom_data:
        print(f"\nFine-tuning on {args.custom_data} …")
        custom_ds = datasets.ImageFolder(args.custom_data,
                                         transform=train_transform)
        n_val = max(1, len(custom_ds) // 5)
        n_train = len(custom_ds) - n_val
        train_ds, val_ds = random_split(custom_ds, [n_train, n_val])
        ft_train = DataLoader(train_ds, batch_size=32, shuffle=True)
        ft_val   = DataLoader(val_ds,   batch_size=32, shuffle=False)
        ft_optim = optim.Adam(model.parameters(), lr=LR * 0.1)
        for epoch in range(1, EPOCHS_FINETUNE + 1):
            tr_loss, tr_acc = train_epoch(model, ft_train, ft_optim, criterion, device)
            va_loss, va_acc = eval_epoch(model, ft_val, criterion, device)
            print(f"  FT Epoch {epoch}/{EPOCHS_FINETUNE}  "
                  f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
