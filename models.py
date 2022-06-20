from matplotlib.pyplot import isinteractive
from sklearn.neighbors import KNeighborsTransformer
import torch
from torch import nn, optim
from torch.nn import Identity as Id, BatchNorm1d, Sequential, LeakyReLU, Conv1d
from math import ceil
import torch.nn.functional as F


def module_name(v):
    return str(v).split('\n')[0]


def module_dfs(v):
    sons = list(v.children())
    return module_name(v) + ','.join(module_dfs(u) for u in sons) + ')' if len(sons) else str(v)


def module_description(model):
    return ''.join(module_dfs(model))


class Activation(LeakyReLU):
    def __init__(self, negative_slope=0.2):
        super().__init__(negative_slope)


class CausalConv(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, shift=0, bias=True):
        super().__init__(in_channels, out_channels,
                         kernel_size, dilation=dilation, bias=bias)
        self.shift = shift
        self.use_bias = bias
        self.padding_shape = (dilation * (kernel_size - 1) + shift, -shift)

    def forward(self, x):
        return super().forward(F.pad(x, self.padding_shape))

    def extra_repr(self):
        return f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size[0]}, dilation={self.dilation[0]}, shift={self.shift}, bias={self.use_bias}'

class TypeI(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, shift=0):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.act = Activation()
        self.conv = CausalConv(in_channels, out_channels, 2, dilation, shift, False)
    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class TypeII(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, shift=0):
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.conv_sigma = CausalConv(in_channels, out_channels, 2, dilation, shift, False)
        self.conv_tanh = CausalConv(in_channels, out_channels, 2, dilation, shift, False)

    def forward(self, x):
        x = self.bn(x)
        a = torch.sigmoid(self.conv_sigma(x))
        b = torch.tanh(self.conv_tanh(x))
        return a * b

class TypeIII(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        assert in_channels == out_channels
        super().__init__()
        self.bn = BatchNorm1d(in_channels)
        self.conv_sigma = CausalConv(in_channels, out_channels, 2, dilation, 0, False)
        self.conv_tanh = CausalConv(in_channels, out_channels, 2, dilation, 0, False)

    def forward(self, x):
        t = self.bn(x)
        a = torch.sigmoid(self.conv_sigma(t))
        b = torch.tanh(self.conv_tanh(t))
        return x + a * b

# class Sum(nn.Module):
#     def __init__(self, f, g):
#         super().__init__()
#         self.f = f
#         self.g = g
#     def forward(self, x):
#         return self.f(x) + self.g(x)


# def ConvBlock(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride, bias=False):
#     return Sequential(
#             BatchNorm2d(in_channels),
#             Activation(),
#             conv_type(in_channels, out_channels, in_size, out_size, kernel_size, stride, bias)
#         )

# def ResSimple(conv_type, channels, kernel_size, stride):
#     return Sum(
#         Id(),
#         Sequential(
#             ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride),
#             ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride)
#         )
#     )

# def ResLearned(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride):
#     mid_channels = round((in_channels * out_channels) ** 0.5)
#     mid_size = round((in_size * out_size) ** 0.5)
#     return Sum(
#         Sequential(
#             BatchNorm2d(in_channels),
#             conv_type(in_channels, out_channels, in_size, out_size, stride ** 2, stride ** 2, bias=False)
#         ),
#         Sequential(
#             ConvBlock(conv_type, in_channels, mid_channels, in_size, mid_size, kernel_size, stride),
#             ConvBlock(conv_type, mid_channels, out_channels, mid_size, out_size, kernel_size, stride)
#         )
#     )

# class VAE(nn.Module):
#     def __init__(self, encoder, mu_head, ls_head, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.mu_head = mu_head
#         self.ls_head = ls_head
#         self.decoder = decoder
#         self.register_buffer('last_z', None)
#         self.register_buffer('z_mean', None)
#         self.register_buffer('z_m2', None)
#         self.register_buffer('z_std', None)
#         self.register_buffer('z_cnt', None)

#     def note_z(self, z):
#         z = z[0:1].detach()
#         self.last_z = z
#         if self.z_cnt is None:
#             self.z_mean = z
#             self.z_m2 = torch.zeros_like(z)
#             self.z_std = torch.zeros_like(z)
#             self.z_cnt = torch.tensor(1.0)
#         else:
#             # update from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#             self.z_cnt += 1
#             delta = z - self.z_mean
#             self.z_mean += delta / self.z_cnt
#             delta2 = z - self.z_mean
#             self.z_m2 += delta * delta2
#             self.z_std = torch.sqrt(self.z_m2 / self.z_cnt)

#     def encode(self, x):
#         t = self.encoder(x)
#         mu, ls = self.mu_head(t), self.ls_head(t)
#         self.note_z(mu)
#         return mu, ls

#     def sample(self, mu, logsigma):
#         if logsigma is None:
#             return mu
#         assert mu.shape == logsigma.shape
#         return mu + torch.randn_like(mu) * logsigma

#     def decode(self, mu, logsigma):
#         z = self.sample(mu, logsigma)
#         return self.decoder(z)

#     def generate_seen(self):
#         self.eval()
#         return self.decode(self.last_z, None)

#     def generate(self):
#         self.eval()
#         return self.decode(self.z_mean, self.z_std)
