import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    BatchNorm1d,
    ConstantPad1d,
    Conv1d,
    Identity,
    LeakyReLU,
    Sequential,
    Sigmoid,
    Tanh,
)


def module_name(v):
    return str(v).split("\n", maxsplit=1)[0]


def module_dfs(v):
    sons = list(v.children())
    return (
        module_name(v) + ",".join(module_dfs(u) for u in sons) + ")" if sons else str(v)
    )


def module_description(model):
    return "".join(module_dfs(model))


class Activation(LeakyReLU):
    def __init__(self, negative_slope=0.2):
        super().__init__(negative_slope)

def Padded(padding, module):
    return Sequential(
        ConstantPad1d(padding, 0),
        module
    )

class Product(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f, self.g = f, g

    def forward(self, x):
        return self.f(x) * self.g(x)


class Sum(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f, self.g = f, g

    def forward(self, x):
        return self.f(x) + self.g(x)


def CausalConv(in_channels, out_channels, kernel_size, dilation, shift=0):
    return Padded(
        (dilation * (kernel_size - 1) + shift, -shift),
        Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=False,
        ),
    )


def ConvBlock(in_channels, out_channels, dilation, shift=0):
    return Sequential(
        BatchNorm1d(in_channels),
        Activation(),
        CausalConv(in_channels, out_channels, 2, dilation, shift),
    )


def GatedConvBlock(in_channels, out_channels, dilation, shift=0):
    return Product(
        Sequential(ConvBlock(in_channels, out_channels, dilation, shift), Tanh()),
        Sequential(ConvBlock(in_channels, out_channels, dilation, shift), Sigmoid()),
    )


def Res(f):
    return Sum(Identity(), f)
