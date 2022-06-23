import torch
from torch import nn
from torch.nn import BatchNorm1d, Sequential, LeakyReLU, Conv1d, Tanh, Sigmoid, Identity
import torch.nn.functional as F


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


class Padded(nn.Module):
    def __init__(self, padding, f):
        super().__init__()
        self.padding = padding
        self.f = f

    def forward(self, x):
        return self.f(F.pad(x, self.padding))

    def extra_repr(self):
        return f"padding={self.padding}"


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
