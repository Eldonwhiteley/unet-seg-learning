"""
Loss functions and evaluation metrics for semantic segmentation.

In image segmentation, Cross-Entropy / BCE calculates pixel-by-pixel cross entropy,
which works well but can suffer when the background dominates the foreground.
Dice Loss optimizes global spatial region overlap (IoU). Combining both provides
superior boundary sharpness and training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coeff(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes the Dice coefficient (F1 score) between predictions and targets.
    Expects probability predictions in [0, 1].
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        total = pred_binary.sum() + target.sum()
        dice = (2.0 * intersection + epsilon) / (total + epsilon)
        return float(dice.item())


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes Intersection over Union (Jaccard index).
    Expects probability predictions in [0, 1].
    """
    with torch.no_grad():
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection
        iou = (intersection + epsilon) / (union + epsilon)
        return float(iou.item())


class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary segmentation.
    Expects raw logits as inputs (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss + Soft Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
