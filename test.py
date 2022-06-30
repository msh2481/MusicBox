import unittest
from time import time

import torch

import build
from models import (
    Activation,
    AdaptiveBN,
    BatchNorm1d,
    CausalConv,
    ConstantPad1d,
    Conv1d,
    ConvBlock,
    GatedConvBlock,
    GroupNet,
    Identity,
    LeakyReLU,
    Linear,
    Module,
    ModuleList,
    Padded,
    Product,
    Res,
    Sequential,
    ShuffleNet,
    Sigmoid,
    SkipConnected,
    Sum,
    Tanh,
    WaveNet,
    module_description,
)
from mu_law import mu_decode, mu_encode
from train import *


class DataLoaders(unittest.TestCase):
    def testV6(self):
        loader = build.dataloader(data="dataset_v6", sample_length=100, batch_size=20)
        self.assertEqual(len(loader), 50)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (20, 256, 100))
        self.assertNotEqual(y.min(), y.max())
        t0 = time()
        for batch in loader:
            pass
        dur = time() - t0
        self.assertLess(dur, 0.5)


class Representation(unittest.TestCase):
    def help(self, model):
        desc = module_description(model)
        model2 = eval(desc)
        desc2 = module_description(model2)
        self.assertEqual(desc, desc2)

    def testPadded(self):
        model = Padded((1, 1), Identity())
        self.help(model)

    def testCausalConv(self):
        model = CausalConv(1, 1, 1, 1)
        self.help(model)

    def testConvBlock(self):
        model = ConvBlock(1, 1, 1)
        self.help(model)

    def testSum(self):
        model = Sum(Identity(), ConvBlock(1, 1, 1))
        self.help(model)

    def testProduct(self):
        model = Product(Identity(), ConvBlock(1, 1, 1))
        self.help(model)

    def testRes(self):
        model = Res(ConvBlock(1, 1, 1))
        self.help(model)
        opt = torch.optim.Adam(model.parameters(), lr=1)
        x = torch.randn((16, 1, 10))
        y = model(x)
        self.assertTrue(torch.allclose(x, y))
        y.sum().backward()
        opt.step()
        y = model(x)
        self.assertFalse(torch.allclose(x, y))

    def testAdaptiveBN(self):
        model = AdaptiveBN(10)
        self.help(model)

    def testWaveNet(self):
        model = WaveNet(3, 2, 10, 20, 30, 40)
        self.help(model)

    def testGroupNet(self):
        model = GroupNet(3, 2, 10, 20, 30, 40, 1, 1, 1)
        self.help(model)

    def testShuffleNet(self):
        model = ShuffleNet(3, 2, 10, 20, 30, 40, 1, 1, 1)
        self.help(model)

class Models(unittest.TestCase):
    def testCausalConv(self):
        model = Sequential(
            CausalConv(1, 10, 2, 1, shift=1),
            Activation(),
            CausalConv(10, 10, 2, 2),
            Activation(),
            CausalConv(10, 10, 2, 4),
            Activation(),
            CausalConv(10, 1, 2, 8),
        )
        x = torch.randn((64, 1, 10))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testConvBlock(self):
        model = Sequential(
            ConvBlock(1, 10, 1, shift=1),
            ConvBlock(10, 10, 2),
            ConvBlock(10, 1, 4),
        )
        x = torch.randn((64, 1, 10))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testSkipConnected(self):
        model = Sequential(
            ConvBlock(1, 10, 1, shift=1),
            SkipConnected(
                ConvBlock(10, 10, 2),
                ConvBlock(10, 10, 4),
            ),
            ConvBlock(10, 1, 1, shift=1),
        )
        x = torch.randn((64, 1, 10))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testGatedConvBlock(self):
        model = GatedConvBlock(256, 256, 2)
        x = torch.randn((64, 256, 1000))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testAdaptiveBN(self):
        model = Sequential(
            CausalConv(256, 256, 1, 1), AdaptiveBN(256), CausalConv(256, 256, 1, 1)
        )
        x = torch.randn((64, 256, 10))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testWaveNet(self):
        model = WaveNet(3, 2, 10, 20, 30, 40)
        x = torch.randn((64, 40, 1000))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testGroupNet(self):
        model = GroupNet(3, 2, 10, 20, 30, 40, 1, 1, 1)
        x = torch.randn((64, 40, 1000))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

    def testShuffleNet(self):
        model = ShuffleNet(3, 2, 10, 20, 30, 40, 1, 1, 1)
        x = torch.randn((64, 40, 1000))
        y = model(x)
        self.assertEqual(y.shape, x.shape)

class MuLaw(unittest.TestCase):
    def testEncodeDecode(self):
        x = torch.randn(10)
        y = mu_encode(x)
        self.assertEqual(y.shape, x.shape)
        x2 = mu_decode(y)
        self.assertEqual(x2.shape, x.shape)
        self.assertTrue(torch.isclose(x, x2).all())