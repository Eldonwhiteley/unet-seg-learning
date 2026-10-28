"""
PyTorch Implementation of the U-Net Architecture.

Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015. https://arxiv.org/abs/1505.04597

Key differences between PyTorch U-Net and TensorFlow/Keras U-Net:
1. Tensor layout:
   - TensorFlow: [Batch, Height, Width, Channels] (NHWC)
   - PyTorch:    [Batch, Channels, Height, Width] (NCHW)
2. Layer definition:
   - TensorFlow often builds models imperatively via the Functional API:
     `c1 = layers.Conv2D(...)(x)`
   - PyTorch models inherit from `nn.Module` and encapsulate reusable blocks
     (`DoubleConv`, `Down`, `Up`) with explicit `forward()` methods.
3. Skip connections:
   - TensorFlow: `layers.concatenate([up, skip], axis=-1)` (along channel axis)
   - PyTorch:    `torch.cat([skip, up], dim=1)` (along channel dimension 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    [ Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU ]
    Standard double convolution building block of U-Net.
    Padding is set to 1 so that spatial dimensions (H, W) remain unchanged.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Contracting (Encoder) block:
    [ MaxPool2d(2x2) -> DoubleConv ]
    Halves spatial dimensions (H, W) and increases channel depth.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Expanding (Decoder) block:
    1. Upsamples lower feature map (ConvTranspose2d or Bilinear interpolation).
    2. Handles spatial padding if dimensions differ by 1 pixel due to rounding.
    3. Concatenates the skip connection from the corresponding encoder stage.
    4. Applies DoubleConv to blend the features.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution doubles H and W and halves channels
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 1. Upsample
        x = self.up(x)

        # 2. Check and pad if H or W differ between x and skip
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )

        # 3. Concatenate along channel dimension (dim=1)
        x = torch.cat([skip, x], dim=1)

        # 4. Blend features with double convolution
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output projection layer:
    1x1 Conv that maps internal feature channels to desired `out_channels`
    (e.g., 1 for binary segmentation or C for C classes).
    Returns raw logits (numerically stable for BCEWithLogitsLoss / CrossEntropyLoss).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.

    Encoder:
      - inc   (in_channels -> base_channels)
      - down1 (base_channels -> base_channels * 2)
      - down2 (base_channels * 2 -> base_channels * 4)
      - down3 (base_channels * 4 -> base_channels * 8)
    Decoder:
      - up1   (base_channels * 8 -> base_channels * 4) + skip from down2
      - up2   (base_channels * 4 -> base_channels * 2) + skip from down1
      - up3   (base_channels * 2 -> base_channels)     + skip from inc
      - outc  (base_channels -> out_channels)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input image channels (e.g. 3 for RGB).
        out_channels : int
            Number of output channels (1 for binary logits, >1 for multi-class).
        base_channels : int
            Initial filter depth (default 32 for fast training, original paper used 64).
        bilinear : bool
            Whether to use bilinear upsampling instead of transposed conv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)

        factor = 2 if bilinear else 1
        self.up1 = Up(b * 8, (b * 4) // factor, bilinear)
        self.up2 = Up(b * 4, (b * 2) // factor, bilinear)
        self.up3 = Up(b * 2, b, bilinear)
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Contracting path (Encoder)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Expanding path (Decoder) with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # 1x1 Conv output logits
        logits = self.outc(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
