import torch
from torch import nn, optim

class ConvDummy(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
    def encode(self, x):
        return self.conv(x), None
    def decode(self, x, ignore):
        return self.conv(x)

def conv1dblock(channels_list, kernel_size, transpose):
    result = []
    for x, y in zip(channels_list, channels_list[1:]):
        if transpose:
            result.append(nn.ConvTranspose1d(x, y, kernel_size=kernel_size, padding=1, bias=False))
        else:
            result.append(nn.Conv1d(x, y, kernel_size=kernel_size, padding=1, bias=False))
        result.append(nn.BatchNorm1d(y))
        result.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*result[:-2])

class Conv1dAE(nn.Module):
    def __init__(self, channels_list, kernel_size):
        super().__init__()
        self.encoder = conv1dblock(channels_list, kernel_size, False)
        self.decoder = conv1dblock(channels_list[::-1], kernel_size, True)
    def encode(self, x):
        return self.encoder(x), None
    def decode(self, z, ignore):
        return self.decoder(z)

class Conv1dVAE(nn.Module):
    def __init__(self, channels_list, kernel_size):
        super().__init__()
        self.encoder = conv1dblock(channels_list[:-1], kernel_size, False)
        self.mu_head = conv1dblock(channels_list[-2:], kernel_size, False)
        self.ls_head = conv1dblock(channels_list[-2:], kernel_size, False)
        self.decoder = conv1dblock(channels_list[::-1], kernel_size, True)
    def encode(self, x):
        z = self.encoder(x)
        return self.mu_head(z), self.ls_head(z)
    def sample(self, mu, logsigma):
        assert mu.shape == logsigma.shape
        return mu + torch.randn_like(mu) * logsigma
    def decode(self, mu, logsigma):
        z = self.sample(mu, logsigma)
        return self.decoder(z)