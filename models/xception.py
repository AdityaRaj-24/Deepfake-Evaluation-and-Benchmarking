"""
xception.py

PyTorch implementation of an Xception-based deepfake detector.

Uses the official timm Xception backbone with a custom binary
classification head.

Compatible with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    """
    Xception detector.

    Input
    -----
    (B, 3, H, W)

    Output
    ------
    (B, 1) raw logits
    """

    def __init__(
        self,
        pretrained=False,
        dropout=0.5,
    ):
        super().__init__()

        # Remove original classifier
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

        self._initialize_classifier()

    def forward(self, x):

        features = self.backbone(x)

        logits = self.classifier(features)

        return logits

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def _initialize_classifier(self):

        for m in self.classifier.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":

    model = XceptionDetector(pretrained=False)

    x = torch.randn(2, 3, 256, 256)

    y = model(x)

    print(model)

    print(f"\nOutput shape : {y.shape}")

    print(f"Parameters   : {model.num_parameters():,}")