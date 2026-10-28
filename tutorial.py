#!/usr/bin/env python3
"""
=============================================================================
🎓 U-Net Semantic Segmentation Tutorial in PyTorch
   Problem: Segmenting Synthetic Colored Squares
=============================================================================

This tutorial guides you step-by-step through:
1. Synthetic Data Generation: Creating random colored squares and ground-truth masks.
2. Tensor Conventions: Explaining the critical difference between TensorFlow (NHWC)
   and PyTorch (NCHW).
3. The U-Net Architecture: Encoder, Bottleneck, Decoder, and Skip Connections.
4. Loss Functions: Why combining BCE and Soft Dice loss beats BCE alone.
5. Training & Evaluation: A rapid training run and visualizing predictions.

Run this script anytime:
    python tutorial.py
=============================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure local src/ is accessible
sys.path.insert(0, os.path.abspath("src"))

from unet_seg.dataset import ColoredSquaresDataset, generate_single_sample
from unet_seg.model import UNet, count_parameters
from unet_seg.loss import BCEDiceLoss, dice_coeff, iou_score
from unet_seg.visualize import plot_predictions


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step_1_data_generation():
    print_header("Step 1: Synthetic Data Generation (Colored Squares)")
    print("""
Why synthetic squares?
----------------------
In real-world computer vision, datasets can be gigabytes in size and have messy
boundaries or label noise. For learning U-Net fundamentals, generating colored
squares on-the-fly provides:
  • Immediate execution with 0 MB dataset download.
  • Pixel-perfect mathematical ground truth masks.
  • Rapid model verification (training converges in seconds).
""")

    raw_img, raw_mask = generate_single_sample(img_size=128)
    print(f"• Generated raw NumPy image shape: {raw_img.shape} (Height, Width, Channels)")
    print(f"• Generated raw NumPy mask shape:  {raw_mask.shape} (Height, Width)")
    print(f"• Image pixel value range:         [{raw_img.min():.2f}, {raw_img.max():.2f}]")
    print(f"• Mask unique class labels:        {list(set(raw_mask.flatten()))}")

    # Inspect PyTorch Dataset tensor conversion
    dataset = ColoredSquaresDataset(num_samples=10, img_size=128)
    img_tensor, mask_tensor = dataset[0]
    print(f"\n• PyTorch Dataset Image Tensor:   {tuple(img_tensor.shape)} (Channels, Height, Width)")
    print(f"• PyTorch Dataset Mask Tensor:    {tuple(mask_tensor.shape)} (Channels, Height, Width)")
    print("  👉 Notice: PyTorch uses (C, H, W), whereas TensorFlow uses (H, W, C)!")


def step_2_unet_architecture(device: torch.device):
    print_header("Step 2: Understanding the U-Net Architecture")
    print("""
U-Net Anatomy:
--------------
                  Input Image [3, 128, 128]
                              │
                      [ DoubleConv ] ─── Skip 1 ───┐
                              ▼                    │
                         [ Down 1 ] ──── Skip 2 ───┼────────┐
                              ▼                    │        │
                         [ Down 2 ] ──── Skip 3 ───┼───┐    │
                              ▼                    │   │    │
                         [ Down 3 ] (Bottleneck)   │   │    │
                              ▼                    │   │    │
                          [ Up 1 ] ◄───────────────┘   │    │
                              ▼                        │    │
                          [ Up 2 ] ◄───────────────────┘    │
                              ▼                             │
                          [ Up 3 ] ◄────────────────────────┘
                              ▼
                        [ 1x1 Conv ]
                              ▼
                     Output Mask [1, 128, 128]

Why Skip Connections are Essential:
-----------------------------------
Downsampling (pooling) captures "WHAT" is in the image (context/semantics),
but loses high-resolution spatial information about "WHERE" edges are.
Skip connections directly copy high-resolution feature maps from encoder to decoder,
giving the network the exact pixel boundaries of the squares!
""")

    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)
    print(f"• Total Trainable Parameters: {count_parameters(model):,}")

    dummy_input = torch.randn(2, 3, 128, 128, device=device)
    dummy_output = model(dummy_input)
    print(f"• Forward pass sanity check:")
    print(f"  Input  Shape: {tuple(dummy_input.shape)}  [Batch, Channels, Height, Width]")
    print(f"  Output Shape: {tuple(dummy_output.shape)} [Batch, Classes, Height, Width]")
    return model


def step_3_loss_function():
    print_header("Step 3: Segmentation Loss (BCE + Soft Dice)")
    print("""
Why not just Binary Cross-Entropy (BCE)?
---------------------------------------
In segmentation, the background usually covers most of the image (class imbalance).
If a model predicts 100% background, BCE loss might still look acceptable!
Dice Loss directly optimizes the Dice Coefficient (F1-score / overlap).
By combining:
    Loss = 0.5 * BCE + 0.5 * DiceLoss
we get both smooth pixel-level gradient propagation and global shape alignment.
""")


def step_4_training(model: nn.Module, device: torch.device):
    print_header("Step 4: Training the U-Net on Colored Squares")

    # Hyperparameters
    EPOCHS = 5
    BATCH_SIZE = 16
    LR = 1e-3

    print(f"• Creating training dataset (500 samples) and validation dataset (100 samples)...")
    train_dataset = ColoredSquaresDataset(num_samples=500, img_size=128, seed=42)
    val_dataset = ColoredSquaresDataset(num_samples=100, img_size=128, seed=999)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"• Training for {EPOCHS} epochs on {device}...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Val Dice':^10} | {'Val IoU':^10}")
    print("-" * 65)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                val_dice += dice_coeff(probs, masks) * images.size(0)
                val_iou += iou_score(probs, masks) * images.size(0)

        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)
        val_iou /= len(val_dataset)
        elapsed = time.time() - t0

        print(
            f"{epoch:^7d} | {train_loss:^12.4f} | {val_loss:^12.4f} | "
            f"{val_dice:^10.4f} | {val_iou:^10.4f} ({elapsed:.1f}s)"
        )

    print("-" * 65)
    print(f"✨ Training complete in {time.time() - start_time:.1f} seconds!")
    print(f"🎯 Final Validation Dice Coefficient: {val_dice:.4f} (1.0 is perfect overlap)")
    return val_dataset


def step_5_visualization(model: nn.Module, dataset: ColoredSquaresDataset, device: torch.device):
    print_header("Step 5: Visualizing Predictions")
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tutorial_results.png")

    plot_predictions(
        model=model,
        dataset=dataset,
        device=device,
        num_samples=4,
        save_path=out_path,
    )
    print(f"\n🎉 Finished! You can inspect the visual results at: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  🚀 Welcome to the PyTorch U-Net Segmentation Tutorial!")
    print(f"     Operating Device: {device}")
    print("=" * 70)

    step_1_data_generation()
    model = step_2_unet_architecture(device)
    step_3_loss_function()
    val_dataset = step_4_training(model, device)
    step_5_visualization(model, val_dataset, device)

    # Save trained tutorial weights
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/best_unet.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"💾 Checkpoint saved to: {ckpt_path}\n")


if __name__ == "__main__":
    main()
