import unittest
from time import time

import torch

import build
from mu_law import mu_encode, mu_decode
from models import (
    Activation,
    BatchNorm1d,
    CausalConv,
    ConstantPad1d,
    Conv1d,
    ConvBlock,
    GatedConvBlock,
    Identity,
    Padded,
    Product,
    Res,
    Sequential,
    Sum,
    module_description,
)
from train import *


class Datasets(unittest.TestCase):
    def testMNIST(self):
        data = build.dataset("mnist")
        self.assertEqual(len(data), 60000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(isinstance(y, int))

    def testV2(self):
        data = build.dataset("dataset_v2")
        self.assertEqual(len(data), 100)
        x, y = data[0]
        self.assertEqual(x.shape, (128, 1025))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)
        self.assertEqual(y, 0)

    def testV3(self):
        data = build.dataset("dataset_v3")
        self.assertEqual(len(data), 1000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 128, 1024))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(y.item(), 0)

    def testV4(self):
        data = build.dataset("dataset_v4")
        self.assertEqual(len(data), 1000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 256, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-30, x.min().item())
        self.assertLessEqual(x.max().item(), 30)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(y.item(), 0)

    def testV5(self):
        data = build.dataset("dataset_v5")
        self.assertEqual(len(data), 1000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 2**16))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-6, x.min().item())
        self.assertLessEqual(x.max().item(), 6)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(y.item(), 0)


class DataLoaders(unittest.TestCase):
    def testMNISTLoader(self):
        loader = build.dataloader(
            data="mnist", batch_size=128, num_workers=0, trash=(1, 2)
        )
        self.assertEqual(len(loader), 469)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (128, 1, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)

    def testV2(self):
        loader = build.dataloader(data="dataset_v2", batch_size=4)
        self.assertEqual(len(loader), 25)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (4, 128, 1025))
        self.assertEqual(x.dtype, torch.float32)
        t0 = time()
        for batch in loader:
            pass
        dur = time() - t0
        self.assertLess(dur, 0.1)

    def testV3(self):
        loader = build.dataloader(data="dataset_v3", batch_size=64)
        self.assertEqual(len(loader), 16)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (64, 1, 128, 1024))
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
        model = CausalConv(1, 1, 1, 0)
        self.help(model)

    def testConvBlock(self):
        model = ConvBlock(1, 1, 1, 0)
        self.help(model)

    def testSum(self):
        model = Sum(Identity(), ConvBlock(1, 1, 1, 0))
        self.help(model)

    def testProduct(self):
        model = Product(Identity(), ConvBlock(1, 1, 1, 0))
        self.help(model)

    def testRes(self):
        model = Res(ConvBlock(1, 1, 1, 0))
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
        print(model)

class MuLaw(unittest.TestCase):
    def testEncodeDecode(self):
        x = torch.randn(10)
        y = mu_encode(x)
        self.assertEqual(y.shape, x.shape)
        x2 = mu_decode(y)
        self.assertEqual(x2.shape, x.shape)
        self.assertTrue(torch.isclose(x, x2).all())