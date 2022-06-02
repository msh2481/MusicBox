import unittest
import build
import torch

class Datasets(unittest.TestCase):
    def testMNIST(self):
        data = build.dataset('mnist')
        self.assertEqual(len(data), 60000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 28, 28))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(type(y) is int)

    def testV2(self):
        data = build.dataset('dataset_v2')
        self.assertEqual(len(data), 100)
        x = data[0]
        self.assertEqual(x.shape, (128, 1025))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)

class DataLoaders(unittest.TestCase):
    def testMNISTLoader(self):
        loader = build.dataloader('mnist', 128, 2)
        self.assertEqual(len(loader), 469)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (128, 1, 28, 28))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)

    def testV2(self):
        loader = build.dataloader('dataset_v2', 4, 2)
        self.assertEqual(len(loader), 25)
        x = next(iter(loader))
        self.assertEqual(x.shape, (4, 128, 1025))
        self.assertEqual(x.dtype, torch.float32)


class ModelOptimSched(unittest.TestCase):
    def testDummy(self):
        m, o, s = build.model_optim_sched(
            'cpu', 'ConvDummy(128)', 'opt.Adam(m.parameters())', 'sch.ExponentialLR(o, 1.0)')
        m.train()
        x = torch.randn((4, 128, 1025))
        y = m(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        o.zero_grad()
        y.sum().backward()
        o.step()
        s.step()


if __name__ == '__main__':
    unittest.main()
