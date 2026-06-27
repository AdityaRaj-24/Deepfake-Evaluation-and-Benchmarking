"""
detector.py

Multiple-Attention Deepfake Detector.

Architecture
------------
Input Image
      │
      ▼
ResNet18 Backbone
      │
      ▼
Multiple Attention Block
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout
      │
      ▼
Fully Connected
      │
      ▼
Binary Logit

Compatible with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn

from .backbone import ResNet18Backbone
from .attention import MultipleAttentionBlock


class MultipleAttentionDetector(nn.Module):
    """
    Multiple-Attention detector for binary deepfake classification.
    """

    def __init__(
        self,
        pretrained=False,
        dropout=0.5,
    ):

        super().__init__()

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        self.backbone = ResNet18Backbone(
            pretrained=pretrained,
        )

        channels = self.backbone.output_channels

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        self.attention = MultipleAttentionBlock(
            channels=channels,
        )

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(dropout),

            nn.Linear(
                channels,
                1,
            ),
        )

        self._initialize_classifier()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, x):

        # Feature extraction
        features = self.backbone(x)

        # Attention refinement
        features = self.attention(features)

        # Global pooling
        features = self.pool(features)

        # Classification
        logits = self.classifier(features)

        return logits

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

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

    model = MultipleAttentionDetector(
        pretrained=False,
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    y = model(x)

    print(model)

    print(f"\nInput shape      : {x.shape}")
    print(f"Output shape     : {y.shape}")
    print(f"Parameters       : {model.num_parameters():,}")
