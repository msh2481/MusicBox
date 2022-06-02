import torch
from torch import nn, optim

class ConvDummy(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1, groups=channels)
    def encode(self, x):
        return self.conv(x)
    def decode(self, x):
        return self.conv(x)
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x