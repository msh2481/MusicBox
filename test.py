import unittest
from unittest.mock import patch
from io import StringIO
import build
import torch
import cProfile
from time import time
from itertools import islice, cycle
import optuna

SKIP_WORKING = True
SKIP_MANUAL = True 

@unittest.skipIf(SKIP_WORKING, '')
class Datasets(unittest.TestCase):
    def testMNIST(self):
        data = build.dataset('mnist')
        self.assertEqual(len(data), 60000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertTrue(type(y) is int)

    def testV2(self):
        data = build.dataset('dataset_v2')
        self.assertEqual(len(data), 100)
        x, y = data[0]
        self.assertEqual(x.shape, (128, 1025))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)
        self.assertEqual(y, 0)

    def testOverfit(self):
        data = build.dataset('dataset_v2_overfit')
        self.assertEqual(len(data), 1)
        x, y = data[0]
        self.assertEqual(x.shape, (128, 11))
        self.assertEqual(y, 0)
    
    def testV3(self):
        data = build.dataset('dataset_v3')
        self.assertEqual(len(data), 1000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 128, 1024))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(y.item(), 0)

    def testV4(self):
        data = build.dataset('dataset_v4')
        self.assertEqual(len(data), 1000)
        x, y = data[0]
        self.assertEqual(x.shape, (1, 256, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-4, x.min().item())
        self.assertLessEqual(x.max().item(), 4)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual(y.item(), 0)

@unittest.skipIf(SKIP_WORKING, '')
class DataLoaders(unittest.TestCase):
    def testMNISTLoader(self):
        loader = build.dataloader(
            data='mnist', batch_size=128, num_workers=0, trash=(1, 2))
        self.assertEqual(len(loader), 469)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (128, 1, 32, 32))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)

    def testV2(self):
        loader = build.dataloader(
            data='dataset_v2', batch_size=4)
        self.assertEqual(len(loader), 25)
        x, y = next(iter(loader))
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

    def testV3(self):
        loader = build.dataloader(
            data='dataset_v3', batch_size=64)
        self.assertEqual(len(loader), 16)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (64, 1, 128, 1024))
        self.assertNotEqual(y.min(), y.max())
        t0 = time()
        for batch in loader:
            pass
        dur = time() - t0
        self.assertLess(dur, 0.5)


class Misc(unittest.TestCase):
    def testSomething(self):
        pass

if __name__ == '__main__':
    unittest.main()