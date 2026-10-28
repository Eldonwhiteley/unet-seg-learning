"""
Visualization script for inspecting U-Net segmentation predictions.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from unet_seg.dataset import ColoredSquaresDataset
from unet_seg.model import UNet


def plot_predictions(
    model: torch.nn.Module,
    dataset: ColoredSquaresDataset,
    device: torch.device,
    num_samples: int = 5,
    save_path: str = "output/predictions.png",
):
    """
    Plots a grid of sample inputs, ground truth masks, predicted heatmaps, and thresholded masks.
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Input Image (RGB)",
        "Ground Truth Mask",
        "Predicted Heatmap (Sigmoid)",
        "Binary Mask (Threshold > 0.5)",
    ]

    with torch.no_grad():
        for i in range(num_samples):
            img_tensor, mask_tensor = dataset[i]
            input_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(input_batch)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred_mask = (probs > 0.5).astype(np.float32)

            # PyTorch CHW -> HWC for Matplotlib
            img_hwc = img_tensor.permute(1, 2, 0).cpu().numpy()
            mask_hw = mask_tensor.squeeze().cpu().numpy()

            # Column 0: Original image
            axes[i, 0].imshow(img_hwc)
            # Column 1: Ground truth mask
            axes[i, 1].imshow(mask_hw, cmap="gray", vmin=0, vmax=1)
            # Column 2: Continuous probability heatmap
            im2 = axes[i, 2].imshow(probs, cmap="magma", vmin=0, vmax=1)
            # Column 3: Binary thresholded prediction
            axes[i, 3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)

            for col in range(4):
                if i == 0:
                    axes[i, col].set_title(col_titles[col], fontsize=12, pad=8)
                axes[i, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize U-Net Predictions")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_unet.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output", type=str, default="output/predictions.png", help="Output plot path"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, base_channels=32).to(device)

    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f" Loaded model checkpoint from {args.checkpoint}")
    else:
        print(f"⚠️ Warning: Checkpoint '{args.checkpoint}' not found. Using untrained model.")

    test_dataset = ColoredSquaresDataset(
        num_samples=max(10, args.num_samples), img_size=128, seed=999
    )
    plot_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        num_samples=args.num_samples,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
