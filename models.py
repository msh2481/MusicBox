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


def channel_shuffle(x, groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083.
    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, length = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, length)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, length)
    return x


class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)


def concat_mish(x):
    assert x.dim() == 3
    return F.mish(torch.cat((x, -x), dim=1))


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
        assert end_groups == 1
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

        self.start_conv = CausalConv(classes, residual_channels, 1, 1, 1, shift=1)
        self.gate = ModuleList()
        self.residual_shuffle = ChannelShuffle(residual_groups)
        self.res = ModuleList()
        self.bn = ModuleList()
        for block, layer in product(range(blocks), range(layers)):
            self.gate.append(
                CausalConv(
                    residual_channels,
                    residual_channels // 2,
                    2,
                    2**layers,
                    gate_groups,
                )
            )
            self.res.append(
                CausalConv(residual_channels, residual_channels, 1, 1, residual_groups)
            )
            self.bn.append(BatchNorm1d(residual_channels))

        self.end_bn1 = BatchNorm1d(residual_channels)
        self.end_conv1 = CausalConv(
            2 * residual_channels, end_channels // 2, 1, 1, end_groups
        )
        self.end_bn2 = BatchNorm1d(end_channels // 2)
        self.end_conv2 = CausalConv(end_channels, classes, 1, 1, end_groups)

    def forward(self, x):
        x = self.start_conv(x)
        for gate, res, bn in zip(self.gate, self.res, self.bn):
            x0 = x
            x = concat_mish(gate(x))
            x = self.residual_shuffle(x)
            x = bn(res(x))
            x = self.residual_shuffle(x)
            x = x0 + x
        x = concat_mish(self.end_bn1(x))
        x = concat_mish(self.end_bn2(self.end_conv1(x)))
        return self.end_conv2(x)

    def alt_repr(self):
        return f"ShuffleNet({self.layers}, {self.blocks}, {self.residual_channels}, {self.skip_channels}, {self.end_channels}, {self.classes}, {self.gate_groups}, {self.residual_groups}, {self.skip_groups}, {self.end_groups})"


# model = ShuffleNet(10, 4, 256, 1, 256, 256, 16, 16, 16, 1)
# print(sum(e.numel() for e in model.parameters()))
