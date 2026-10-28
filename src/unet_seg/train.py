"""
Training script for U-Net segmentation.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unet_seg.dataset import ColoredSquaresDataset
from unet_seg.model import UNet, count_parameters
from unet_seg.loss import BCEDiceLoss, dice_coeff, iou_score


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Trains model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluates model on validation dataset, returns (val_loss, dice, iou)."""
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            total_dice += dice_coeff(probs, masks) * images.size(0)
            total_iou += iou_score(probs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on Colored Squares")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=128, help="Image dimension")
    parser.add_argument("--train-samples", type=int, default=600, help="Train samples")
    parser.add_argument("--val-samples", type=int, default=150, help="Val samples")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save dir")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Datasets and Loaders
    print("📦 Generating synthetic datasets...")
    train_dataset = ColoredSquaresDataset(
        num_samples=args.train_samples,
        img_size=args.img_size,
        seed=42,
    )
    val_dataset = ColoredSquaresDataset(
        num_samples=args.val_samples,
        img_size=args.img_size,
        seed=1337,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    print(f"🧠 U-Net initialized with {count_parameters(model):,} parameters")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(args.save_dir, "best_unet.pth")

    print("\n🏁 Starting training:")
    print("-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Val Dice':^10} | {'Val IoU':^10}")
    print("-" * 65)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        epoch_sec = time.time() - t0

        print(
            f"{epoch:^7d} | {train_loss:^12.4f} | {val_loss:^12.4f} | "
            f"{val_dice:^10.4f} | {val_iou:^10.4f} ({epoch_sec:.1f}s)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                },
                best_checkpoint_path,
            )

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"✅ Training completed in {total_time:.1f}s!")
    print(f"💾 Best model checkpoint saved to: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
