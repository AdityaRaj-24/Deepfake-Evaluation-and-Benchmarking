"""
meso4.py

PyTorch implementation of Meso4
(Afchar et al., MesoNet)

Returns raw logits.
Train using:
    criterion = nn.BCEWithLogitsLoss()
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool"""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x):
        return self.block(x)


class Meso4(nn.Module):
    """
    Meso4 architecture.

    Expected input:
        (B, 3, 256, 256)

    Output:
        (B, 1) raw logits
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 8, 3, pool_size=2),
            ConvBlock(8, 8, 5, pool_size=2),
            ConvBlock(8, 16, 5, pool_size=2),
            ConvBlock(16, 16, 5, pool_size=4),
        )

        # Feature map:
        # 256 -> 128 -> 64 -> 32 -> 8
        # Shape = (16, 8, 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(16 * 8 * 8, 16, bias=True),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(16, 1, bias=True),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def _initialize_weights(self):
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    model = Meso4()

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print(model)
    print(f"\nOutput shape: {y.shape}")
    print(f"Trainable parameters: {model.num_parameters():,}")