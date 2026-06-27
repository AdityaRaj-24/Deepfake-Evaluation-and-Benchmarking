"""
patch_resnet.py

Patch-based ResNet detector for deepfake detection.

The image is divided into non-overlapping patches.
All patches are processed together through a shared ResNet18
backbone, and the patch-level predictions are averaged to obtain
the final image prediction.

Compatible with BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class PatchResNet(nn.Module):
    """
    Patch-based ResNet detector.

    Input
    -----
        (B,3,H,W)

    Output
    ------
        (B,1) logits
    """

    def __init__(
        self,
        patch_size=128,
        pretrained=False,
        dropout=0.5,
    ):
        super().__init__()

        self.patch_size = patch_size

        # --------------------------------------------------
        # Backbone
        # --------------------------------------------------

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(
            weights=weights,
        )

        feature_dim = self.backbone.fc.in_features

        # remove original classifier
        self.backbone.fc = nn.Identity()

        # Image-level classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

        self._initialize_classifier()

    # --------------------------------------------------
    # Patch extraction
    # --------------------------------------------------

    def extract_patches(self, x):
        """
        Converts

            (B,C,H,W)

        into

            (B*N,C,P,P)

        where N is the number of patches.
        """

        B, C, H, W = x.shape
        P = self.patch_size

        if H < P or W < P:
            raise ValueError(
                "Patch size cannot exceed image dimensions."
            )

        if H % P != 0 or W % P != 0:
            raise ValueError(
                "Image size must be divisible by patch size."
            )

        patches = (
            x.unfold(2, P, P)
             .unfold(3, P, P)
             .permute(0, 2, 3, 1, 4, 5)
             .contiguous()
        )

        n_h = patches.shape[1]
        n_w = patches.shape[2]

        patches = patches.view(
            B * n_h * n_w,
            C,
            P,
            P,
        )

        return patches, n_h * n_w

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
    
    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        logits : (B, 1)
        """

        batch_size = x.size(0)

        # Extract all patches from the batch
        patches, patches_per_image = self.extract_patches(x)

        # --------------------------------------------------
        # Single forward pass through ResNet
        # Shape:
        #   (B * N, 3, P, P)
        # --------------------------------------------------

        # --------------------------------------------------
        # Regroup patch features
        #
        # (B*N,F)
        #   ↓
        # (B,N,F)
        # --------------------------------------------------
        
        features = self.backbone(patches)

        features = features.view(
            batch_size,
            patches_per_image,
            -1,
        )

        # Average patch embeddings to obtain
        # one feature vector per image.
        features = features.mean(dim=1)

        image_logits = self.classifier(features)

        return image_logits


if __name__ == "__main__":

    model = PatchResNet(
        patch_size=128,
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

    print(f"\nOutput shape : {y.shape}")

    print(f"Parameters   : {model.num_parameters():,}")

    patches, n = model.extract_patches(x)

    grid_h = x.shape[2] // model.patch_size
    grid_w = x.shape[3] // model.patch_size

    print(f"Patches/image : {n}")
    print(f"Patch grid    : {grid_h} × {grid_w}")
    print(f"Patch tensor  : {patches.shape}")