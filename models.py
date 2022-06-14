from sklearn.neighbors import KNeighborsTransformer
import torch
from torch import nn, optim


class ConvDummy(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Conv1d(
            channels, channels, kernel_size=1, padding='same', groups=channels, bias=False)
        self.decoder = nn.Conv1d(
            channels, channels, kernel_size=1, padding='same', groups=channels, bias=False)

    def encode(self, x):
        return self.encoder(x), None

    def decode(self, x, ignore):
        return self.decoder(x)


def conv1dblock(channels_list, kernel_size, transpose):
    assert kernel_size % 2 == 1
    result = []
    for x, y in zip(channels_list, channels_list[1:]):
        if transpose:
            result.append(nn.ConvTranspose1d(
                x, y, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
        else:
            result.append(nn.Conv1d(x, y, kernel_size=kernel_size,
                          padding=kernel_size//2, bias=False))
        result.append(nn.BatchNorm1d(y))
        result.append(nn.LeakyReLU(0.2))
    return result[:-2]

class Generatable:
    def generate(self):
        self.eval()
        device = next(iter(self.decoder.parameters())).device
        z = torch.randn(self.z_shape, device=device)
        return self.decode(z.unsqueeze(0), None).squeeze(0)

class Conv1dAE(nn.Module, Generatable):
    def __init__(self, channels_list, kernel_size):
        super().__init__()
        self.z_shape = (channels_list[-1], 1025)
        self.encoder = nn.Sequential(
            *conv1dblock(channels_list, kernel_size, False))
        self.decoder = nn.Sequential(
            *conv1dblock(channels_list[::-1], kernel_size, True))

    def encode(self, x):
        return self.encoder(x), None

    def decode(self, z, ignore):
        return self.decoder(z)

class Conv1dVAE(nn.Module, Generatable):
    def __init__(self, channels_list, kernel_size):
        super().__init__()
        self.z_shape = (channels_list[-1], 1025)
        self.encoder = nn.Sequential(
            *conv1dblock(channels_list[:-1], kernel_size, False))
        self.mu_head = nn.Sequential(
            *conv1dblock(channels_list[-2:], kernel_size, False))
        self.ls_head = nn.Sequential(
            *conv1dblock(channels_list[-2:], kernel_size, False))
        self.decoder = nn.Sequential(
            *conv1dblock(channels_list[::-1], kernel_size, True))

    def encode(self, x):
        z = self.encoder(x)
        return self.mu_head(z), self.ls_head(z)

    def sample(self, mu, logsigma):
        if logsigma is None:
            return mu
        assert mu.shape == logsigma.shape
        return mu + torch.randn_like(mu) * logsigma

    def decode(self, mu, logsigma):
        z = self.sample(mu, logsigma)
        return self.decoder(z)


def conv2dblock(channels_list, strides_list, kernel_sizes, transpose):
    assert len(strides_list) == len(channels_list) - 1
    assert len(kernel_sizes) == len(channels_list) - 1
    result = []
    for i in range(len(channels_list) - 1):
        stride = strides_list[i]
        kernel_size = kernel_sizes[i]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        x, y = channels_list[i], channels_list[i + 1]
        assert all(sz % 2 == 1 for sz in kernel_size)
        padding = tuple([sz // 2 for sz in kernel_size])
        if transpose:
            output_padding = tuple([stride[i] - kernel_size[i] % 2 for i in range(len(kernel_size))])
            result.append(nn.ConvTranspose2d(x, y, kernel_size=kernel_size, stride=stride,
                          padding=padding, output_padding=output_padding, bias=False))
        else:
            result.append(nn.Conv2d(x, y, kernel_size=kernel_size,
                          padding=padding, stride=stride, bias=False))
        result.append(nn.BatchNorm2d(y))
        result.append(nn.LeakyReLU(0.2))
    return result[:-2]


class Conv2dVAE(nn.Module, Generatable):
    def __init__(self, channels_list, strides_list, kernel_sizes):
        super().__init__()
        self.z_shape = (channels_list[-1], 1, 1)
        self.encoder = nn.Sequential(
            *conv2dblock(channels_list[:-1], strides_list[:-1], kernel_sizes[:-1], False))
        self.mu_head = nn.Sequential(
            *conv2dblock(channels_list[-2:], strides_list[-1:], kernel_sizes[-1:], False))
        self.ls_head = nn.Sequential(
            *conv2dblock(channels_list[-2:], strides_list[-1:], kernel_sizes[-1:], False))
        self.decoder = nn.Sequential(
            *conv2dblock(channels_list[::-1], strides_list[::-1], kernel_sizes[::-1], True))

    def encode(self, x):
        z = self.encoder(x)
        return self.mu_head(z), self.ls_head(z)

    def sample(self, mu, logsigma):
        if logsigma is None:
            return mu
        assert mu.shape == logsigma.shape
        return mu + torch.randn_like(mu) * logsigma

    def decode(self, mu, logsigma):
        z = self.sample(mu, logsigma)
        return self.decoder(z)

class ResEncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 256, 256, 0)
        # self.bn2 = nn.BatchNorm2d(c)
        # self.conv2 = nn.Conv2d(c, c, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2)
        # self.pool = nn.AvgPool2d(4)
    def forward(self, x):
        # f = self.pool(x)
        g = self.conv1(self.act(self.bn1(x)))
        # g = self.conv1(self.act(self.bn2(g)))
        return g

class ResDecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.ConvTranspose2d(c, c, 256, 256, 0)
        # self.bn2 = nn.BatchNorm2d(c)
        # self.conv2 = nn.ConvTranspose2d(c, c, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2)
        # self.pool = nn.Upsample(scale_factor=4)
    def forward(self, x):
        # f = self.pool(x)
        g = self.conv1(self.act(self.bn1(x)))
        # g = self.conv2(self.act(self.bn2(g)))
        return g

class M1(nn.Module, Generatable):
    def __init__(self, ch, dim):
        super().__init__()
        self.z_shape = (dim, 1, 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, 1, 1),
            ResEncoderBlock(ch),
        )
        self.mu_head = nn.Conv2d(ch, dim, 1)
        self.ls_head = nn.Conv2d(ch, dim, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, ch, 1),
            ResDecoderBlock(ch),
            nn.ConvTranspose2d(ch, 1, 3, 1, 1)
        )

    def encode(self, x):
        z = self.encoder(x)
        return self.mu_head(z), self.ls_head(z)

    def sample(self, mu, logsigma):
        if logsigma is None:
            return mu
        assert mu.shape == logsigma.shape
        return mu + torch.randn_like(mu) * logsigma

    def decode(self, mu, logsigma):
        z = self.sample(mu, logsigma)
        return self.decoder(z)