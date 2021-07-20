import torch
from torch import nn


class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.conv(3, 8),  # 14
            nn.ReLU(),
            self.maxpool(),
            self.conv(8, 8),  # 7
            nn.ReLU(),
            self.maxpool(),
            self.conv(8, 1),  # 7
            self.maxpool(),
            nn.Flatten(),  # remove (1,1) grid
            nn.Linear(100, 256, bias=True),
            #                         nn.Linear(400, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2, bias=True)
        )

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def conv(ni,nf):
        return nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def maxpool():
        return nn.MaxPool2d(kernel_size=2, stride=2)
