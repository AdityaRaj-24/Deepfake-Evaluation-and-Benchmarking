"""
attention.py

Attention modules for the Multiple-Attention Deepfake Detector.

Contains:
    - SpatialAttention
    - ChannelAttention
    - MultipleAttentionBlock
"""

import torch
import torch.nn as nn


# --------------------------------------------------------
# Spatial Attention
# --------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    Learns a spatial attention map highlighting
    important image regions.
    """

    def __init__(self, channels):

        super().__init__()

        self.attention = nn.Sequential(

            nn.Conv2d(
                channels,
                channels // 2,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels // 2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        attention = self.attention(x)

        return x * attention


# --------------------------------------------------------
# Channel Attention
# --------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Learns channel-wise importance using
    squeeze-and-excitation.
    """

    def __init__(
        self,
        channels,
        reduction=16,
    ):

        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(
                channels,
                hidden,
                bias=False,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden,
                channels,
                bias=False,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        b, c, _, _ = x.shape

        weights = self.pool(x).view(b, c)

        weights = self.fc(weights)

        weights = weights.view(b, c, 1, 1)

        return x * weights


# --------------------------------------------------------
# Multiple Attention Block
# --------------------------------------------------------

class MultipleAttentionBlock(nn.Module):
    """
    Combines multiple attention mechanisms.

    Output =
        Spatial Attention +
        Channel Attention
    """

    def __init__(
        self,
        channels,
    ):

        super().__init__()

        self.spatial = SpatialAttention(
            channels,
        )

        self.channel = ChannelAttention(
            channels,
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        spatial_features = self.spatial(x)

        channel_features = self.channel(x)

        fused = torch.cat(
            [
                spatial_features,
                channel_features,
            ],
            dim=1,
        )

        return self.fusion(fused)


# --------------------------------------------------------
# Testing
# --------------------------------------------------------

if __name__ == "__main__":

    x = torch.randn(
        4,
        512,
        8,
        8,
    )

    block = MultipleAttentionBlock(
        channels=512,
    )

    y = block(x)

    print(block)

    print(f"\nInput shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    params = sum(
        p.numel()
        for p in block.parameters()
        if p.requires_grad
    )

    print(f"Parameters   : {params:,}")