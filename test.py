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
    Identity,
    LeakyReLU,
    Linear,
    MixtureNet,
    Module,
    ModuleList,
    Padded,
    Product,
    QueueNet,
    Res,
    Sequential,
    Sigmoid,
    SkipConnected,
    Sum,
    Tanh,
    dilate,
    module_description,
)
from mu_law import mu_decode, mu_encode


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

    def testQueueNet(self):
        model = QueueNet()
        self.help(model)

    def testMixtureNet(self):
        model = MixtureNet()
        self.help(model)

class Models(unittest.TestCase):
    def testCausalConv(self):
        model = Sequential(
            CausalConv(1, 10, 2, 1),
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
            ConvBlock(1, 10, 1),
            ConvBlock(10, 10, 2),
            ConvBlock(10, 1, 4),
        )
        x = torch.randn((64, 1, 10))

        y = model(x)

        self.assertEqual(y.shape, x.shape)

    def testSkipConnected(self):
        model = Sequential(
            ConvBlock(1, 10, 1),
            SkipConnected(
                ConvBlock(10, 10, 2),
                ConvBlock(10, 10, 4),
            ),
            ConvBlock(10, 1, 1),
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

    def testDilateShape(self):
        x = torch.randn((10 * 64, 256, 1000))

        y = dilate(x, 128, 64)

        self.assertEqual(y.shape, (10 * 128, 256, 500))

    def testDilateDown(self):
        x = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])

        y = dilate(x, 2, 1)

        self.assertTrue(torch.equal(y, torch.tensor([[[1, 3, 5, 7, 9]], [[2, 4, 6, 8, 10]]])))
    
    def testDilateUp(self):
        x = torch.tensor([[[1, 3, 5, 7, 9]], [[2, 4, 6, 8, 10]]])

        y = dilate(x, 1, 2)

        self.assertTrue(torch.equal(y, torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])))
    
    def testDilateDownWithPadding(self):
        x = torch.tensor([[[1, 2], [3, 4]]])

        y = dilate(x, 2, 1)

        self.assertTrue(torch.equal(y, torch.tensor([[[1], [3]], [[2], [4]]])))

    def testFastForward(self):
        model = QueueNet()
        x = torch.randn((10, 256, 2**13))
        
        y = model(x)

        self.assertEqual(y.shape, (10, 256, 2**13))

    def testFastGenerateShape(self):
        model = QueueNet()
        model.reset()
        x = torch.zeros((256,))

        for i in range(10):
            y = model.generate(x)
            self.assertEqual(y.shape, (256,))

    def testFastGenerateCorrectness(self):
        model = QueueNet(layers=3, blocks=2)
        model.reset()
        h = torch.zeros((256, 1))

        for i in range(10):
            self.assertEqual(h.shape, (256, 1 + i))
            y_fast = model.generate(h[:, -1])
            y_slow = model(h.unsqueeze(0))[0, :, -1]
            self.assertTrue(torch.allclose(y_fast, y_slow, rtol=1e-3))
            h = torch.cat((h, y_fast.unsqueeze(1)), dim=1)
    
    def testMixtureFastGenerateCorrectness(self):
        model = MixtureNet(layers=3, blocks=2)
        model.reset()
        h = torch.zeros((256, 1))

        for i in range(10):
            self.assertEqual(h.shape, (256, 1 + i))
            y_fast = model.generate(h[:, -1])
            y_slow = model(h.unsqueeze(0))[0, :, -1]
            self.assertTrue(torch.allclose(y_fast, y_slow, rtol=1e-3))
            h = torch.cat((h, y_fast.unsqueeze(1)), dim=1)

class MuLaw(unittest.TestCase):
    def testEncodeDecode(self):
        x = torch.randn(10)
        y = mu_encode(x)
        self.assertEqual(y.shape, x.shape)
        x2 = mu_decode(y)
        self.assertEqual(x2.shape, x.shape)
        self.assertTrue(torch.isclose(x, x2).all())