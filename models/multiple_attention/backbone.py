"""
backbone.py

Backbone network for the Multiple-Attention Deepfake Detector.

Uses ResNet18 as the feature extractor and returns
the final convolutional feature map.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    """
    ResNet18 feature extractor.

    Input
    -----
        (B,3,H,W)

    Output
    ------
        (B,512,H/32,W/32)
    """

    def __init__(self, pretrained=False):

        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet18(
            weights=weights,
        )

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x

    @property
    def output_channels(self):
        """
        Number of channels in the final feature map.
        """
        return 512

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":

    model = ResNet18Backbone(
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
    print(f"Feature map      : {y.shape}")
    print(f"Output channels  : {model.output_channels}")
    print(f"Parameters       : {model.num_parameters():,}")