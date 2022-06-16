from sklearn.neighbors import KNeighborsTransformer
import torch
from torch import nn, optim
from torch.nn import Identity as Id, BatchNorm2d as BN2d, Sequential as Seq, AvgPool2d as Avg2d
from math import ceil 

# stride * (out_size - 1) = in_size + 2 * padding - dilation * (kernel_size - 1)
def calculate_padding(in_size, out_size, kernel_size, stride):
    double_padding = stride * (out_size - 1) - in_size + kernel_size
    if double_padding % 2:
        raise Exception(f"Can't find integer padding for {in_size} -> {out_size} with kernel {kernel_size} and stride {stride}")
    return double_padding // 2

def C2d(in_channels, out_channels, in_size, out_size, kernel_size, stride, bias=True):
    padding = calculate_padding(in_size, out_size, kernel_size, stride)
    assert out_size == (in_size + 2 * padding - (kernel_size - 1) + stride - 1) // stride
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

def CT2d(in_channels, out_channels, in_size, out_size, kernel_size, stride, bias=True):
    padding = calculate_padding(in_size, out_size, kernel_size, stride)
    assert out_size == (in_size + 2 * padding - (kernel_size - 1) + stride - 1) // stride
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

class Sum(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
    def forward(self, x):
        return self.f(x) + self.g(x)

def Activation():
    return nn.LeakyReLU(0.2)

def ConvBlock(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride, bias=False):
    return Seq(
            BN2d(in_channels),
            Activation(),
            conv_type(in_channels, out_channels, in_size, out_size, kernel_size, stride, bias)
        )

def ResSimple(conv_type, channels, kernel_size, stride):
    return Sum(
        Id(),
        Seq(
            ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride),
            ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride)
        )
    )

def ResLearned(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride):
    mid_channels = round((in_channels * out_channels) ** 0.5)
    mid_size = round((in_size * out_size) ** 0.5)
    return Sum(
        Seq(
            BN2d(in_channels),
            conv_type(in_channels, out_channels, in_size, out_size, kernel_size, stride ** 2, bias=False)
        ),
        Seq(
            ConvBlock(conv_type, in_channels, mid_channels, in_size, mid_size, kernel_size, stride),
            ConvBlock(conv_type, mid_channels, out_channels, mid_size, out_size, kernel_size, stride)
        )
    )

def generate(model):
    model.eval()
    device = next(iter(model.decoder.parameters())).device
    z = torch.randn(model.z_shape, device=device)
    return model.decode(z.unsqueeze(0), None).squeeze(0)