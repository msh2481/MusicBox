import unittest
from unittest.mock import patch
from io import StringIO
import build
import torch
import cProfile
from time import time
from itertools import islice, cycle
import optuna
from models import *
from train import trainVAE

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

class Description(unittest.TestCase):
    def testBackAndForth(self):
        c = 16
        encoder = Sequential(
            ConvBlock(C2d, 1, c, 256, 64, 4, 4),
            ConvBlock(C2d, c, c, 64, 16, 4, 4),
            ConvBlock(C2d, c, c, 16, 4, 4, 4),
        )
        mu_head = ConvBlock(C2d, c, c, 4, 1, 4, 4)
        ls_head = ConvBlock(C2d, c, c, 4, 1, 4, 4)
        decoder = Sequential(
            ConvBlock(CT2d, c, c, 4, 1, 4, 4),
            ConvBlock(CT2d, c, c, 16, 4, 4, 4),
            ConvBlock(CT2d, c, c, 64, 16, 4, 4),
            ConvBlock(CT2d, c, 1, 256, 64, 4, 4),
        )
        vae = VAE(encoder, mu_head, ls_head, decoder)
        self.assertEqual(module_description(eval(module_description(vae))), module_description(vae))

class Train(unittest.TestCase):
    model_loader = 'VAE(Sequential(Sequential(BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),Conv2d(1, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),Conv2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),Conv2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False))),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),Conv2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),Conv2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=(4, 4), bias=False)),Sequential(BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),LeakyReLU(negative_slope=0.2),ConvTranspose2d(16, 1, kernel_size=(4, 4), stride=(4, 4), bias=False))))'
    loader = build.dataloader(data='dataset_v4_overfit', batch_size=16)
    model, optim, sched = build.model_optim_sched(device='cpu', model_loader=model_loader, optim_loader='opt.AdamW(m.parameters(), lr=1e-3)')
    logger = build.logger(console=True, save_rate=0.0, sample_rate=0.0)
    criterion = build.criterion(k_mse=1.0, k_kl=None)
    trainVAE(trial=None, device='cpu', loader=loader, model=model, optim=optim, sched=sched, criterion=criterion, logger=logger, epochs=3)
    print(model.generate().shape)

if __name__ == '__main__':
    unittest.main()