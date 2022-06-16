from sklearn.neighbors import KNeighborsTransformer
import torch
from torch import nn, optim
from torch.nn import Identity as Id, BatchNorm2d, Sequential, LeakyReLU, Conv2d, ConvTranspose2d
from math import ceil 

def module_name(v):
    return str(v).split('\n')[0]
def module_dfs(v):
    sons = list(v.children())
    return module_name(v) + ','.join(module_dfs(u) for u in sons) + ')' if len(sons) else str(v)
def module_description(model):
    return ''.join(module_dfs(model))

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
    return LeakyReLU(0.2)

def ConvBlock(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride, bias=False):
    return Sequential(
            BatchNorm2d(in_channels),
            Activation(),
            conv_type(in_channels, out_channels, in_size, out_size, kernel_size, stride, bias)
        )

def ResSimple(conv_type, channels, kernel_size, stride):
    return Sum(
        Id(),
        Sequential(
            ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride),
            ConvBlock(conv_type, channels, channels, kernel_size, kernel_size, kernel_size, stride)
        )
    )

def ResLearned(conv_type, in_channels, out_channels, in_size, out_size, kernel_size, stride):
    mid_channels = round((in_channels * out_channels) ** 0.5)
    mid_size = round((in_size * out_size) ** 0.5)
    return Sum(
        Sequential(
            BatchNorm2d(in_channels),
            conv_type(in_channels, out_channels, in_size, out_size, kernel_size, stride ** 2, bias=False)
        ),
        Sequential(
            ConvBlock(conv_type, in_channels, mid_channels, in_size, mid_size, kernel_size, stride),
            ConvBlock(conv_type, mid_channels, out_channels, mid_size, out_size, kernel_size, stride)
        )
    )

class VAE(nn.Module):
    def __init__(self, encoder, mu_head, ls_head, decoder):
        super().__init__()
        self.encoder = encoder
        self.mu_head = mu_head
        self.ls_head = ls_head
        self.decoder = decoder
    
        self.last_z = None
        self.z_mean = 0
        self.z_std = 0
        self.z_cnt = 0
    
    def note_z(self, z):
        z = z[0:1]
        self.last_z = z
        self.z_mean = (self.z_mean * self.z_cnt + z) / (self.z_cnt + 1)
        self.z_std = (self.z_std * self.z_cnt + z) / (self.z_cnt + 1)
        self.z_cnt += 1

    def encode(self, x):
        t = self.encoder(x)
        mu, ls = self.mu_head(t), self.ls_head(t)
        self.note_z(mu)
        return mu, ls
    
    def sample(self, mu, logsigma):
        if logsigma is None:
            return mu
        assert mu.shape == logsigma.shape
        return mu + torch.randn_like(mu) * logsigma

    def decode(self, mu, logsigma):
        z = self.sample(mu, logsigma)
        return self.decoder(z)

    def generate_seen(self):
        self.eval()
        return self.decode(self.last_z, None)
    
    def generate(self):
        self.eval()
        return self.decode(self.z_mean, self.z_std)