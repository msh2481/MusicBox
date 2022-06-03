import unittest
from unittest.mock import patch
from io import StringIO
import build
import torch
import cProfile
from time import time
from itertools import islice, cycle

SKIP_WORKING = True

@unittest.skipIf(SKIP_WORKING, '')
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

    def testOverfit(self):
        data = build.dataset('dataset_v2_overfit')
        self.assertEqual(len(data), 1)
        x = data[0]
        self.assertEqual(x.shape, (128, 11))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)

@unittest.skipIf(SKIP_WORKING, '')
class DataLoaders(unittest.TestCase):
    def testMNISTLoader(self):
        loader = build.dataloader(
            data='mnist', batch_size=128, num_workers=0, trash=(1, 2))
        self.assertEqual(len(loader), 469)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (128, 1, 28, 28))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)
    
    def testV2(self):
        loader = build.dataloader(
            data='dataset_v2', batch_size=4)
        self.assertEqual(len(loader), 25)
        x = next(iter(loader))
        self.assertEqual(x.shape, (4, 128, 1025))
        self.assertEqual(x.dtype, torch.float32)
        t0 = time()
        for batch in loader:
            pass
        dur = time() - t0
        self.assertLess(dur, 0.1)

    def testOverfit(self):
        loader = build.dataloader(
            data='dataset_v2_overfit', batch_size=1)
        self.assertEqual(len(loader), 1)
        t0 = time()
        for batch in loader:
            pass
        dur = time() - t0
        self.assertLess(dur, 0.01)

@unittest.skipIf(SKIP_WORKING, '')
class ModelOptimSched(unittest.TestCase):
    def testDummy(self):
        m, o, s = build.model_optim_sched(
            device='cpu', model_loader='ConvDummy(128)', optim_loader='opt.Adam(m.parameters())', sched_loader='sch.ExponentialLR(o, 1.0)')
        m.train()
        self.assertEqual(sum(x.numel() for x in m.parameters()), 512)
        x = torch.randn((4, 128, 1025))
        z, aux = m.encode(x)
        self.assertEqual(z.dtype, x.dtype)
        self.assertTrue(aux is None)
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        o.zero_grad()
        y.sum().backward()
        o.step()
        s.step()

    def testConvAE(self):
        m, o, s = build.model_optim_sched(
            device='cpu', model_loader='Conv1dAE([128, 10], 3)', optim_loader='opt.AdamW(m.parameters(), weight_decay=params["wd"])', wd=1e-6)
        m.train()
        x = torch.randn((4, 128, 1025))
        z, aux = m.encode(x)
        self.assertTrue(aux is None)
        self.assertEqual(z.shape, (4, 10, 1025))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.shape, x.shape)

    def testConvVAE(self):
        m, o, s = build.model_optim_sched(
            device='cpu', model_loader='Conv1dVAE([128, 10], 3)', optim_loader='opt.Adam(m.parameters())')
        m.train()
        x = torch.randn((4, 128, 1025))
        z, aux = m.encode(x)
        self.assertEqual(z.shape, aux.shape)
        self.assertEqual(z.shape, (4, 10, 1025))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.shape, x.shape)

# @unittest.skipIf(SKIP_WORKING, '')
class TrainAE(unittest.TestCase):
    @unittest.skipIf(SKIP_WORKING, '')
    def testDummy(self):
        build.run({
            'trainer': 'trainVAE',

            'data': 'dataset_v2',
            'batch_size': 16,
            'epochs': 30,

            'model_loader': 'ConvDummy(128)',
            'optim_loader': 'opt.Adam(m.parameters(), lr=0.1)',
            'sched_loader': 'sch.ExponentialLR(o, 0.95)',

            'k_mse': 1,
            'k_kl': None,

            'console': True,
            'save_rate': None
        })
        # Why not 0?
        # epoch mse
        # 0     1000
        # 10    10
        # 20    0.005
        # 30    4e-6
    
    # @unittest.skipIf(SKIP_WORKING, '')
    def testOverfit(self):
        build.run({
            'trainer': 'trainVAE',

            'data': 'dataset_v2_overfit',
            'batch_size': 10,
            'epochs': 500,

            'model_loader': 'Conv1dAE([128, 10], kernel_size=3)',
            'optim_loader': 'opt.Adam(m.parameters(), lr=0.01)',
            'k_mse': 1,
            'k_kl': None,

            'console': False,
            'save_rate': None
        })
        # mse ~ 0.01 after 500 epochs

if __name__ == '__main__':
    unittest.main()