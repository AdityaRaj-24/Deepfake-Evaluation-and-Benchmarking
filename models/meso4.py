"""
meso4.py

PyTorch implementation of Meso4 (Afchar et al., MesoNet)

Returns raw logits for BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size//2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size,
                         stride=pool_size,
                         padding=1),
        )

    def forward(self, x):
        return self.block(x)


class Meso4(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 8, 3, 2),
            ConvBlock(8, 8, 5, 2),
            ConvBlock(8, 16, 5, 2),
            ConvBlock(16, 16, 5, 4),
        )

        with torch.no_grad():
            dummy = torch.zeros(1,3,256,256)
            flatten_dim = self.features(dummy).reshape(1,-1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flatten_dim,16),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16,1),
        )

    def forward(self,x):
        return self.classifier(self.features(x))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__=="__main__":
    m=Meso4()
    y=m(torch.randn(2,3,256,256))
    print(y.shape)
    print(m.num_parameters())
