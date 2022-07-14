from turtle import st
import unittest
from time import time
from matplotlib.pyplot import cla

import torch
import torch.nn.functional as F

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
    LogisticMixture,
    MixtureNet,
    Module,
    ModuleList,
    Padded,
    Product,
    QueueNet,
    Res,
    Sequential,
    Sigmoid,
    Softmax,
    SkipConnected,
    Sum,
    Tanh,
    dilate,
    discretize,
    module_description,
    nll_without_logits,
)
from mu_law import mu_decode, mu_encode
from sampling import beam_search

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

        self.assertTrue(
            torch.equal(y, torch.tensor([[[1, 3, 5, 7, 9]], [[2, 4, 6, 8, 10]]]))
        )

    def testDilateUp(self):
        x = torch.tensor([[[1, 3, 5, 7, 9]], [[2, 4, 6, 8, 10]]])

        y = dilate(x, 1, 2)

        self.assertTrue(
            torch.equal(y, torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]))
        )

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

    def testFastGenerateCorrectnessSmall(self):
        c = 2
        model = QueueNet(layers=1, blocks=1, classes=2)
        model.reset()
        h = torch.zeros((c, 1))

        for i in range(10):
            self.assertEqual(h.shape, (c, 1 + i))
            y_fast = model.generate(h[:, -1])
            y_slow = model(h.unsqueeze(0))[0, :, -1]
            self.assertTrue(torch.allclose(y_fast, y_slow, rtol=1e-3))
            h = torch.cat((h, y_fast.unsqueeze(1)), dim=1)

    def testFastGenerateCorrectness(self):
        c = 64
        model = QueueNet(layers=3, blocks=2, classes=c)
        model.reset()
        h = torch.zeros((c, 1))

        for i in range(10):
            self.assertEqual(h.shape, (c, 1 + i))
            y_fast = model.generate(h[:, -1])
            y_slow = model(h.unsqueeze(0))[0, :, -1]
            self.assertTrue(torch.allclose(y_fast, y_slow, rtol=1e-3))
            h = torch.cat((h, y_fast.unsqueeze(1)), dim=1)

    def testMixtureFastGenerateCorrectness(self):
        c = 3
        model = MixtureNet(layers=3, blocks=2, classes=c)
        model.reset()
        h = torch.zeros((c, 1))

        for i in range(10):
            self.assertEqual(h.shape, (c, 1 + i))
            y_fast = model.generate(h[:, -1])
            y_slow = model(h.unsqueeze(0))[0, :, -1]
            self.assertTrue(torch.allclose(y_fast, y_slow, atol=0.1))
            h = torch.cat((h, y_fast.unsqueeze(1)), dim=1)

    def testMixtureReturnsDistribution(self):
        batch = 10
        mixtures = 2
        classes = 10
        length = 64
        model = LogisticMixture(mixtures, classes)
        x = torch.randn((batch, 2 * mixtures, length), requires_grad=True)

        y = model(x).sum(dim=1)

        self.assertTrue(torch.isclose(y, torch.tensor(1.)).all())

    def testMixtureGradients(self):
        batch = 10
        mixtures = 5
        classes = 256
        length = 64
        model = LogisticMixture(mixtures, classes)
        x = torch.randn((batch, 2 * mixtures, length), requires_grad=True)
        y = model(x)

        y.sum().backward()

        self.assertFalse(torch.isnan(x.grad).any())

    def testNLL(self):
        batch = 10
        mixtures = 2
        classes = 256
        length = 1000
        f = torch.randint(0, classes, (batch, length))
        y = F.one_hot(f).transpose(1, 2)
        assert y.shape == (batch, classes, length)
        x = torch.randn((batch, 2 * mixtures, length))
        m = LogisticMixture(mixtures, classes)
        p = m(x)
        assert p.shape == (batch, classes, length)

        l = nll_without_logits(p, y).item()

        self.assertLess(l, 10)
        self.assertGreater(l, 1)

class MuLaw(unittest.TestCase):
    def testEncodeDecode(self):
        x = torch.randn(10)
        y = mu_encode(x)
        self.assertEqual(y.shape, x.shape)
        x2 = mu_decode(y)
        self.assertEqual(x2.shape, x.shape)
        self.assertTrue(torch.isclose(x, x2).all())


class AdversarialBeamMock:
    def __init__(self, prefix):
        self.prefix = prefix

    def single(self, x):
        alphabet, length = x.size()
        result = torch.linspace(0, 1, alphabet).view(alphabet, 1)
        if length >= self.prefix:
            k = (x[:, : self.prefix].argmax(dim=0).min() / alphabet + 0.01) ** 3
            result /= k
        return result

    def nobatch(self, x):
        return torch.cat([self.single(x[:, :i]) for i in range(x.size(1) + 1)], dim=1)

    def __call__(self, x):
        if x.dim() == 2:
            return self.nobatch(x)
        return torch.stack([self.nobatch(x[i, :, :]) for i in range(x.size(0))], dim=0)


def beam_search_slow(
    model, alphabet, beam_size, branch_factor, sample_length, temperature, seed
):
    alphabet, seed_length = seed.size()
    beam = torch.stack([seed for i in range(beam_size)], dim=0)
    assert beam.shape == (beam_size, *seed.shape)
    nll = [0] * beam_size
    for iter in range(sample_length - seed.size(1)):
        new_beam = []
        logits = model(beam)[:, :, -1]
        assert logits.shape == (beam_size, alphabet)
        choices = []
        for prev in range(beam_size):
            for branch in range(branch_factor):
                token = torch.multinomial(
                    torch.softmax(logits[prev, :] / temperature, dim=-1), 1
                )
                choices.append(
                    (
                        nll[prev] - torch.log_softmax(logits[prev], dim=0)[token].item(),
                        torch.cat((beam[prev], F.one_hot(token, alphabet).view(alphabet, 1)), dim=1),
                    )
                )

        choices = sorted(choices, key=lambda x: x[0])[:beam_size]

        for i, (p, seq) in enumerate(choices):
            new_beam.append(seq)
            nll[i] = p
        beam = torch.stack(new_beam, dim=0)
    idx = torch.tensor(nll).argmin()
    return beam[idx], nll[idx]

def runSlow(beam_size, branch_factor, return_beam=False):
    torch.manual_seed(0)
    alphabet = 256
    start = F.one_hot(torch.tensor(alphabet - 1), alphabet).view(alphabet, 1)
    beam, nll = beam_search_slow(AdversarialBeamMock(3), alphabet, beam_size, branch_factor, 10, 1.0, start)
    return (beam, nll) if return_beam else nll

def runFast(beam_size, branch_factor, return_beam=False):
    torch.manual_seed(0)
    alphabet = 256
    start = F.one_hot(torch.tensor(alphabet - 1), alphabet).view(alphabet, 1)
    beam, nll = beam_search(AdversarialBeamMock(3), alphabet, beam_size, branch_factor, 10, 1.0, start)
    assert isinstance(nll, float)
    return (beam, nll) if return_beam else nll

class BeamSearch(unittest.TestCase):
    def avg(self, runner, beam, branch):
        k = 7
        return sum(runner(beam, branch) for _ in range(k)) / k
    
    def checkBeamAndBranch(self, runner):
        self.assertAlmostEqual(self.avg(runner, 1, 1), 47., delta=2.0)
        self.assertAlmostEqual(self.avg(runner, 1, 10), 46., delta=2.0)
        self.assertAlmostEqual(self.avg(runner, 10, 1), 21., delta=2.0)
        self.assertAlmostEqual(self.avg(runner, 3, 3), 44., delta=2.0)
    
    def testSlow(self):
        self.checkBeamAndBranch(runSlow)
    
    def testFast(self):
        self.checkBeamAndBranch(runFast)