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
        self.assertEqual(x.shape, (128, 1024))
        self.assertEqual(x.dtype, torch.float32)
        self.assertLessEqual(-100, x.min().item())
        self.assertLessEqual(x.max().item(), 0.1)
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

def just_model(model_loader):
    return build.model_optim_sched(device='cpu', model_loader=model_loader, optim_loader='None', sched_loader='None')[0]

@unittest.skipIf(SKIP_WORKING, '')
class ModelOptimSched(unittest.TestCase):
    def testDummy(self):
        m, o, s = build.model_optim_sched(
            device='cpu', model_loader='ConvDummy(128)', optim_loader='opt.Adam(m.parameters())', sched_loader='sch.ExponentialLR(o, 1.0)')
        m.train()
        self.assertEqual(sum(x.numel() for x in m.parameters()), 256)
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

@unittest.skipIf(SKIP_WORKING, '')
class Models(unittest.TestCase):
    def testConvAE(self):
        m = just_model('Conv1dAE([128, 10], 5)')
        m.train()
        x = torch.randn((4, 128, 1025))
        z, aux = m.encode(x)
        self.assertTrue(aux is None)
        self.assertEqual(z.shape, (4, 10, 1025))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        g = m.generate()
        self.assertEqual(g.shape, x[0].shape)

    def testConvVAE(self):
        m = just_model('Conv1dVAE([128, 10], 3)')
        m.train()
        x = torch.randn((4, 128, 1025))
        z, aux = m.encode(x)
        self.assertEqual(z.shape, aux.shape)
        self.assertEqual(z.shape, (4, 10, 1025))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        g = m.generate()
        self.assertEqual(g.shape, x[0].shape)

    def testConv2dVAE(self):
        m = just_model('Conv2dVAE([1, 10], [(1, 1)], [(7, 3)])')
        m.train()
        x = torch.randn((4, 1, 128, 1024))
        z, aux = m.encode(x)
        self.assertEqual(z.shape, aux.shape)
        self.assertEqual(z.shape, (4, 10, 128, 1024))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        g = m.generate()
        self.assertNotEqual(g.shape, x[0].shape)

    def testConv2dVAEWithStride(self):
        m = just_model('Conv2dVAE([1, 256, 128, 64, 32, 16], [(4, 4), (4, 4), (4, 4), (2, 4), (1, 4)], [5, 5, 5, (3, 5), (1, 5)])')
        m.train()
        x = torch.randn((7, 1, 128, 1024))
        z, aux = m.encode(x)
        self.assertEqual(z.shape, aux.shape)
        self.assertEqual(z.shape, (7, 16, 1, 1))
        y = m.decode(z, aux)
        self.assertEqual(y.shape, x.shape)
        g = m.generate()
        self.assertEqual(g.shape, x[0].shape)

@unittest.skipIf(SKIP_MANUAL, '')
class TrainAE(unittest.TestCase):
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
            'save_rate': None,
            'sample_rate': None
        })
        # Why not 0?
        # epoch mse
        # 0     1000
        # 10    10
        # 20    0.005
        # 30    4e-6
    
    def testOverfit(self):
        build.run({
            'trainer': 'trainVAE',

            'data': 'dataset_v2_overfit',
            'batch_size': 10,
            'epochs': 500,

            'device': 'cpu',
            'model_loader': 'Conv1dAE([128, 10], kernel_size=3)',
            'optim_loader': 'opt.Adam(m.parameters(), lr=0.01)',
            'k_mse': 1,
            'k_kl': None,

            'console': False,
            'save_rate': None,
            'sample_rate': 0.0
        })
        # mse ~ 0.01 after 500 epochs

@unittest.skipIf(SKIP_MANUAL, '')
class Optuna(unittest.TestCase):
    def testDummy(self):
        def objective(trial):
            cfg = {
                'trainer': 'trainVAE',
                'data': 'dataset_v2',
                'epochs': 3,
                'device': 'cpu',
                'optim_loader': 'opt.AdamW(m.parameters(), lr=params["lr"], weight_decay=params["wd"])',
                'k_mse': 1.0,
                'k_kl': None,
                'console': True,
                'save_rate': None
            }
            cfg['batch_size'] = 2**trial.suggest_int('log_bs', 2, 6)
            kernel_size = 1 + 2 * trial.suggest_int('ks/2', 1, 2)
            if trial.suggest_categorical('layers', [1, 2]) == 1:
                layer_1 = trial.suggest_int('layer_1', 8, 32, log=True)
                cfg['model_loader'] = f'Conv1dAE([128, {layer_1}], kernel_size={kernel_size})'
            else:
                layer_1 = trial.suggest_int('layer_1', 32, 64, log=True)
                layer_2 = trial.suggest_int('layer_2', 8, 32, log=True)
                cfg['model_loader'] = f'Conv1dAE([128, {layer_1}, {layer_2}], kernel_size={kernel_size})'

            cfg['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            cfg['wd'] = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
            return build.run(cfg)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=3)
    
    def test2dVAE(self):
        def objective(trial):
            cfg = {
                'trainer': 'trainVAE',
                'data': 'mnist',
                'epochs': 3,
                'device': 'cpu',
                'optim_loader': 'opt.AdamW(m.parameters(), lr=1e-3)',
                'k_mse': 1.0,
                'k_kl': None,
                'console': False,
                'save_rate': 0.0,
                'sample_rate': 0.0
            }
            cfg['batch_size'] = 2**trial.suggest_int('log_bs', 2, 6)
            cfg['kernel_size'] = trial.suggest_int('ks', 3, 5)
            cfg['model_loader'] = f'Conv2dVAE([1, 256, 128, 64, 32, 16], [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)], [3, 3, 3, 3, 3])'
            return build.run(cfg)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=1)

if __name__ == '__main__':
    unittest.main()
