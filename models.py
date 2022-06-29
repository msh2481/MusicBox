from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    BatchNorm1d,
    ConstantPad1d,
    Conv1d,
    Identity,
    LeakyReLU,
    Linear,
    Module,
    ModuleList,
    Parameter,
    ParameterList,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)


def module_name(v):
    return str(v).split("\n", maxsplit=1)[0]


def module_dfs(v):
    if hasattr(v, "alt_repr"):
        return v.alt_repr()
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
    return Sequential(ConstantPad1d(padding, 0), module)


class AdaptiveBN(BatchNorm1d):
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels, **kwargs)
        self.k = Parameter(torch.tensor(0.5))

    def forward(self, x):
        return super().forward(x) * self.k + x * (1 - self.k)


class SkipConnected(nn.Sequential):
    def forward(self, x):
        output = x
        for module in self:
            x = module(x)
            output = output + x
        return output


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
        ),
    )


def ConvBlock(in_channels, out_channels, dilation, shift=0):
    return Sequential(
        Activation(),
        CausalConv(in_channels, out_channels, 2, dilation, shift),
    )


def GatedConvBlock(in_channels, out_channels, dilation, shift=0):
    kernel_size = 2
    return Product(
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, shift), Tanh()
        ),
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, shift),
            Sigmoid(),
        ),
    )


class Res(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.k = Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x + self.k * self.f(x)


class WaveNet(Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        residual_channels=32,
        skip_channels=256,
        end_channels=256,
        classes=256,
    ):
        super().__init__()
        self.layers = layers
        self.blocks = blocks
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.classes = classes

        self.start_conv = CausalConv(classes, residual_channels, 1, 1, shift=1)
        self.gate = ModuleList()
        self.res = ModuleList()
        self.skip = ModuleList()
        self.alpha = ParameterList()
        for block, layer in product(range(blocks), range(layers)):
            self.gate.append(
                GatedConvBlock(residual_channels, residual_channels, 2**layers)
            )
            self.res.append(CausalConv(residual_channels, residual_channels, 1, 1))
            self.skip.append(CausalConv(residual_channels, skip_channels, 1, 1))
            self.alpha.append(Parameter(torch.tensor(0.)))
        self.end_conv1 = CausalConv(residual_channels, end_channels, 1, 1)
        self.end_conv2 = CausalConv(end_channels, classes, 1, 1)

    def forward(self, x):
        x = self.start_conv(x)
        skip_sum = 0
        for alpha, gate, res, skip in zip(self.alpha, self.gate, self.res, self.skip):
            x0 = x
            x = gate(x)
            skip_sum = skip_sum + x
            x = x0 + alpha * res(x)
        x = F.relu(skip_sum)
        x = F.relu(self.end_conv1(x))
        return self.end_conv2(x)

    def alt_repr(self):
        return f"WaveNet({self.layers}, {self.blocks}, {self.residual_channels}, {self.skip_channels}, {self.end_channels}, {self.classes})"
