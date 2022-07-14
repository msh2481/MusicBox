from itertools import product
from matplotlib import pyplot as plt

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
from tqdm import tqdm


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


def CausalConv(in_channels, out_channels, kernel_size, dilation, groups=1):
    return Padded(
        (dilation * (kernel_size - 1), 0),
        Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, groups=groups
        ),
    )


def ConvBlock(in_channels, out_channels, dilation, groups=1):
    return Sequential(
        Activation(),
        CausalConv(in_channels, out_channels, 2, dilation, groups),
    )


def GatedConvBlock(in_channels, out_channels, dilation, groups=1):
    kernel_size = 2
    return Product(
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, groups),
            Tanh(),
        ),
        Sequential(
            CausalConv(in_channels, out_channels, kernel_size, dilation, groups),
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


def dilate(x, dilation, init_dilation, pad_start=True):
    """
    :param x: Tensor of size (batch_size * init_dilation, channels, length)
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [bn, channels, length] = x.size()
    if bn % init_dilation:
        raise RuntimeError(f"bn = {bn}, init_dilation = {init_dilation}")

    if dilation == 1:
        # (b * d, c, l) -> (b, c, l * d)
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(channels, length * init_dilation, bn // init_dilation)
        x = x.permute(2, 0, 1).contiguous()
        return x

    assert dilation % init_dilation == 0
    dilation_factor = dilation // init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = (length + dilation_factor - 1) // dilation_factor * dilation_factor
    if new_l != length:
        x = F.pad(x, (new_l - length, 0) if pad_start else (0, new_l - length))
        length = new_l
    assert length % dilation_factor == 0
    length //= dilation_factor
    bn *= dilation_factor

    # reshape according to dilation
    # -> (channels, old_length, batch_size * init_dilation)
    x = x.permute(1, 2, 0).contiguous()
    # -> (channels, new_length, batch_size * dilation)
    x = x.view(channels, length, bn)
    #  -> (batch_size * dilation, channels, new_length)
    x = x.permute(2, 0, 1).contiguous()

    assert x.size(0) % dilation == 0
    return x


class TensorQueue:
    def __init__(self, length, channels):
        super().__init__()
        self.length = length
        self.channels = channels
        self.reset()

    def reset(self):
        self.data = torch.zeros(self.length, self.channels)
        # the data lies in [out_ptr, in_ptr)
        self.pointer = 0

    def push(self, x):
        self.data[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.length

    def pop(self, dilation):
        assert dilation < self.length
        i = (self.pointer - dilation - 1) % self.length
        j = (self.pointer - 1) % self.length
        return torch.stack((self.data[i], self.data[j]), dim=1)


class QueueNet(Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        res_channels=256,
        end_channels=256,
        classes=256,
        groups=1,
        out_classes=None,
    ):
        super().__init__()
        self.layers = layers
        self.blocks = blocks
        self.res_channels = res_channels
        self.end_channels = end_channels
        self.classes = classes
        self.out_classes = out_classes or classes
        self.groups = groups

        self.start_conv = CausalConv(self.classes, self.res_channels, 1, 1, 1)
        self.shuffle = ChannelShuffle(self.groups)
        self.gate = ModuleList(
            CausalConv(self.res_channels, self.res_channels // 2, 2, 1, self.groups)
            for _ in range(self.layers * self.blocks)
        )
        self.res = ModuleList(
            CausalConv(self.res_channels, self.res_channels, 1, 1, self.groups)
            for _ in range(self.layers * self.blocks)
        )
        self.bn = ModuleList(
            BatchNorm1d(self.res_channels) for _ in range(self.layers * self.blocks)
        )

        self.queues = [
            TensorQueue(2**self.layers + 1, self.res_channels)
            for _ in range(self.layers * self.blocks)
        ]

        self.end_bn1 = BatchNorm1d(self.res_channels)
        self.end_conv1 = CausalConv(
            2 * self.res_channels, self.end_channels // 2, 1, 1, 1
        )
        self.end_bn2 = BatchNorm1d(self.end_channels // 2)
        self.end_conv2 = CausalConv(self.end_channels, self.out_classes, 1, 1, 1)

    def _forward(self, x, dilate_fn):
        x = self.start_conv(x)
        prev_dilation = 1
        for i, (block, layer) in enumerate(
            product(range(self.blocks), range(self.layers))
        ):
            x, prev_dilation = dilate_fn(x, 2**layer, prev_dilation, i), 2**layer
            x0 = x
            x = concat_mish(self.gate[i](x))
            x = self.shuffle(x)
            x = self.bn[i](self.res[i](x))
            x = self.shuffle(x)
            x = x0 + x
        x = concat_mish(self.end_bn1(x))
        x = concat_mish(self.end_bn2(self.end_conv1(x)))
        x = self.end_conv2(x)
        return x

    def forward(self, x):
        def fw_dilate_fn(x, dilation, init_dilation, i):
            return dilate(x, dilation, init_dilation)

        batch_size, channels, length = x.shape
        assert channels == self.classes

        receptive_field = self.blocks * 2**self.layers
        padded_size = max(length, receptive_field)
        mult = 2**self.layers
        padded_size = (padded_size + mult - 1) // mult * mult

        x = F.pad(x, (padded_size - length, 0))
        x = self._forward(x, fw_dilate_fn)
        x = dilate(x, 1, 2 ** (self.layers - 1))

        assert x.shape == (batch_size, self.out_classes, padded_size)
        return x

    def generate(self, x):
        """
        :param x: Tensor of size (1, C, L) with an one-hot encoding of new data point.
        """

        def gen_dilate_fn(x, dilation, init_dilation, i):
            assert x.dim() == 3
            assert x.size(0) == 1
            x = x[0, :, -1]  # or 0?
            self.queues[i].push(x.squeeze())
            x = self.queues[i].pop(dilation).unsqueeze(0)
            assert x.dim() == 3
            assert x.squeeze().dim() == 2  # (C, L)
            return x

        self.eval()
        return self._forward(x.view(1, self.classes, 1), gen_dilate_fn)[0, :, -1]

    def reset(self, zero_init=True, show_progress=True):
        self.eval()
        for queue in self.queues:
            queue.reset()
        if not zero_init:
            return
        receptive_field = self.blocks * 2**self.layers
        progress = (
            tqdm(range(receptive_field), desc="Reset")
            if show_progress
            else range(receptive_field)
        )
        for _ in progress:
            self.generate(torch.zeros(self.classes))

    def alt_repr(self):
        return f"QueueNet(layers={self.layers}, blocks={self.blocks}, res_channels={self.res_channels}, end_channels={self.end_channels}, classes={self.classes}, groups={self.groups})"


def logistic_cdf(x, loc, scale):
    return torch.sigmoid((x - loc) / scale)


def discretize(x, mixtures, bins):
    batch_size, channels, length = x.size()
    assert channels == 3 * mixtures
    points = torch.linspace(0.5, bins - 0.5, bins, device=x.device)
    lb, rb = points - 0.5, points + 0.5
    lb, rb = lb.view(1, bins, 1, 1), rb.view(1, bins, 1, 1)
    locs = bins * (0.5 + x[:, :mixtures, :]).view(batch_size, 1, mixtures, length)
    scales = bins / 10 * torch.exp(x[:, mixtures:2*mixtures, :]).view(batch_size, 1, mixtures, length)
    coefs = x[:, 2*mixtures:, :].view(batch_size, 1, mixtures, length)
    coefs = F.softmax(coefs, dim=2)

    lprobs = logistic_cdf(lb, locs, scales)
    lprobs = torch.cat(
        (torch.zeros_like(lprobs[:, :1, :, :]), lprobs[:, 1:, :, :]), dim=1
    )
    assert lprobs.shape == (batch_size, bins, mixtures, length)
    rprobs = logistic_cdf(rb, locs, scales)
    rprobs = torch.cat(
        (rprobs[:, :-1, :, :], torch.ones_like(rprobs[:, -1:, :, :])), dim=1
    )
    assert rprobs.shape == (batch_size, bins, mixtures, length)
    probs = (coefs * (rprobs - lprobs)).sum(dim=2)
    return probs


class LogisticMixture(Module):
    def __init__(self, mixtures, bins):
        super().__init__()
        self.mixtures = mixtures
        self.bins = bins

    def forward(self, x):
        batch, in_channels, length = x.shape
        assert in_channels == 3 * self.mixtures
        return discretize(x, self.mixtures, self.bins)


class MixtureNet(Module):
    def __init__(
        self,
        layers=10,
        blocks=4,
        res_channels=256,
        end_channels=256,
        classes=256,
        mixtures=5,
        groups=1,
    ):
        super().__init__()
        self.mixtures = mixtures
        self.queue_net = QueueNet(
            layers, blocks, res_channels, end_channels, classes, groups, 3 * mixtures
        )
        self.mixture = LogisticMixture(mixtures, classes)

    def cont_forward(self, x):
        return self.queue_net.forward(x)

    def forward(self, x):
        batch, channels, length = x.shape
        assert channels == self.queue_net.classes
        return self.mixture(self.cont_forward(x).view(batch, 3 * self.mixtures, -1))

    def cont_generate(self, x):
        return self.queue_net.generate(x)

    def generate(self, x):
        (channels,) = x.shape
        assert channels == self.queue_net.classes
        return self.mixture(self.cont_generate(x).view(1, 3 * self.mixtures, 1))[
            0, :, -1
        ]

    def reset(self, zero_init=True, show_progress=True):
        self.queue_net.reset(zero_init, show_progress)

    def alt_repr(self):
        return f"MixtureNet(layers={self.queue_net.layers}, blocks={self.queue_net.blocks}, res_channels={self.queue_net.res_channels}, end_channels={self.queue_net.end_channels}, classes={self.queue_net.classes}, mixtures={self.mixtures}, groups={self.queue_net.groups})"


def nll_without_logits(predict, target):
    eps = 1e-9
    assert predict.shape == target.shape
    batch, channel, length = predict.shape
    return (-torch.log(predict + eps) * target).sum(dim=-2).mean()


# from torchinfo import summary
# m = QueueNet(layers=10, blocks=4, res_channels=256, end_channels=1024, classes=256, groups=1)
# print(summary(m, (1, 256, 2 ** 13)))
