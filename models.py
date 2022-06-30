from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    BatchNorm1d,
    ConstantPad1d,
    Conv1d,
    ChannelShuffle,
    Identity,
    LeakyReLU,
    Linear,
    Mish,
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


def CausalConv(in_channels, out_channels, kernel_size, dilation, groups=1, shift=0):
    return Padded(
        (dilation * (kernel_size - 1) + shift, -shift),
        Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, groups=groups
        ),
    )


def ConvBlock(in_channels, out_channels, dilation, groups=1, shift=0):
    return Sequential(
        Activation(),
        CausalConv(in_channels, out_channels, 2, dilation, groups, shift),
    )


def GatedConvBlock(in_channels, out_channels, dilation, groups=1, shift=0):
    kernel_size = 2
    return Product(
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, groups, shift),
            Tanh(),
        ),
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, groups, shift),
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

        self.start_conv = CausalConv(classes, residual_channels, 1, 1, 1, shift=1)
        self.gate = ModuleList()
        self.res = ModuleList()
        self.bn = ModuleList()
        self.skip = ModuleList()
        for block, layer in product(range(blocks), range(layers)):
            self.gate.append(
                GatedConvBlock(residual_channels, residual_channels, 2**layers, 1)
            )
            self.res.append(CausalConv(residual_channels, residual_channels, 1, 1, 1))
            self.skip.append(CausalConv(residual_channels, skip_channels, 1, 1, 1))
            self.bn.append(BatchNorm1d(residual_channels))
        self.bn1 = BatchNorm1d(residual_channels)
        self.end_conv1 = CausalConv(residual_channels, end_channels, 1, 1, 1)
        self.bn2 = BatchNorm1d(end_channels)
        self.end_conv2 = CausalConv(end_channels, classes, 1, 1, 1)

    def forward(self, x):
        x = self.start_conv(x)
        skip_sum = 0
        for gate, res, skip, bn in zip(self.gate, self.res, self.skip, self.bn):
            x0 = x
            x = gate(x)
            skip_sum = skip_sum + x
            x = x0 + bn(res(x))
        x = F.mish(self.bn1(skip_sum))
        x = F.mish(self.bn2(self.end_conv1(x)))
        return self.end_conv2(x)

    def alt_repr(self):
        return f"WaveNet({self.layers}, {self.blocks}, {self.residual_channels}, {self.skip_channels}, {self.end_channels}, {self.classes})"


class GroupNet(Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        residual_channels=32,
        skip_channels=256,
        end_channels=256,
        classes=256,
        gate_groups=1,
        residual_groups=1,
        skip_groups=1,
        end_groups=1,
    ):
        super().__init__()
        self.layers = layers
        self.blocks = blocks
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.classes = classes
        self.gate_groups = gate_groups
        self.residual_groups = residual_groups
        self.skip_groups = skip_groups
        self.end_groups = end_groups

        self.start_conv = CausalConv(
            classes, residual_channels, 1, 1, end_groups, shift=1
        )
        self.gate = ModuleList()
        self.res = ModuleList()
        self.bn = ModuleList()
        self.skip = ModuleList()
        for block, layer in product(range(blocks), range(layers)):
            self.gate.append(
                GatedConvBlock(
                    residual_channels, residual_channels, 2**layers, gate_groups
                )
            )
            self.res.append(
                CausalConv(residual_channels, residual_channels, 1, 1, residual_groups)
            )
            self.skip.append(
                CausalConv(residual_channels, skip_channels, 1, 1, skip_groups)
            )
            self.bn.append(BatchNorm1d(residual_channels))
        self.bn1 = BatchNorm1d(residual_channels)
        self.end_conv1 = CausalConv(residual_channels, end_channels, 1, 1, end_groups)
        self.bn2 = BatchNorm1d(end_channels)
        self.end_conv2 = CausalConv(end_channels, classes, 1, 1, end_groups)

    def forward(self, x):
        x = self.start_conv(x)
        skip_sum = 0
        for gate, res, skip, bn in zip(self.gate, self.res, self.skip, self.bn):
            x0 = x
            x = gate(x)
            skip_sum = skip_sum + x
            x = x0 + bn(res(x))
        x = F.mish(self.bn1(skip_sum))
        x = F.mish(self.bn2(self.end_conv1(x)))
        return self.end_conv2(x)

    def alt_repr(self):
        return f"GroupNet({self.layers}, {self.blocks}, {self.residual_channels}, {self.skip_channels}, {self.end_channels}, {self.classes}, {self.gate_groups}, {self.residual_groups}, {self.skip_groups}, {self.end_groups})"


class ShuffleNet(Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        residual_channels=32,
        skip_channels=256,
        end_channels=256,
        classes=256,
        gate_groups=1,
        residual_groups=1,
        skip_groups=1,
        end_groups=1,
    ):
        super().__init__()
        self.layers = layers
        self.blocks = blocks
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.classes = classes
        self.gate_groups = gate_groups
        self.residual_groups = residual_groups
        self.skip_groups = skip_groups
        self.end_groups = end_groups

        self.start_conv = CausalConv(
            classes, residual_channels, 1, 1, 1, shift=1
        )
        self.gate = ModuleList()
        self.shuffle = ModuleList()
        self.res = ModuleList()
        self.bn = ModuleList()
        self.skip = ModuleList()
        for block, layer in product(range(blocks), range(layers)):
            self.gate.append(
                GatedConvBlock(
                    residual_channels, residual_channels, 2**layers, gate_groups
                )
            )
            self.shuffle.append(ChannelShuffle(1))
            self.res.append(
                CausalConv(residual_channels, residual_channels, 1, 1, residual_groups)
            )
            self.skip.append(
                CausalConv(residual_channels, skip_channels, 1, 1, skip_groups)
            )
            self.bn.append(BatchNorm1d(residual_channels))
        self.end_convs = Sequential(
            BatchNorm1d(residual_channels),
            Mish(),
            ChannelShuffle(1),
            CausalConv(residual_channels, end_channels, 1, 1, end_groups),
            BatchNorm1d(end_channels),
            Mish(),
            ChannelShuffle(1),
            CausalConv(end_channels, classes, 1, 1, end_groups),
        )

    def forward(self, x):
        x = self.start_conv(x)
        skip_sum = 0
        for gate, shuffle, res, skip, bn in zip(
            self.gate, self.shuffle, self.res, self.skip, self.bn
        ):
            x0 = x
            x = shuffle(gate(x))
            skip_sum = skip_sum + x
            x = x0 + bn(res(x))
        return self.end_convs(x)

    def alt_repr(self):
        return f"ShuffleNet({self.layers}, {self.blocks}, {self.residual_channels}, {self.skip_channels}, {self.end_channels}, {self.classes}, {self.gate_groups}, {self.residual_groups}, {self.skip_groups}, {self.end_groups})"
