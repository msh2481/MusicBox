import unittest
from unittest.mock import patch
from io import StringIO
import build
import torch


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


@unittest.skipIf(SKIP_WORKING, '')
class DataLoaders(unittest.TestCase):
    def testMNISTLoader(self):
        loader = build.dataloader(
            name='mnist', batch_size=128, num_workers=2, trash=(1, 2))
        self.assertEqual(len(loader), 469)
        x, y = next(iter(loader))
        self.assertEqual(x.shape, (128, 1, 28, 28))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.long)

    def testV2(self):
        loader = build.dataloader(
            name='dataset_v2', batch_size=4, num_workers=2)
        self.assertEqual(len(loader), 25)
        x = next(iter(loader))
        self.assertEqual(x.shape, (4, 128, 1025))
        self.assertEqual(x.dtype, torch.float32)

@unittest.skipIf(SKIP_WORKING, '')
class ModelOptimSched(unittest.TestCase):
    def testDummy(self):
        m, o, s = build.model_optim_sched(
            device='cpu', model_loader='ConvDummy(128)', optim_loader='opt.Adam(m.parameters())', sched_loader='sch.ExponentialLR(o, 1.0)')
        m.train()
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


class TrainAE(unittest.TestCase):
    @patch('sys.stdout', new_callable=StringIO)
    def testDummy(self, stdout):
        build.run({
            'trainer': 'trainVAE',

            'data': 'dataset_v2',
            'batch_size': 8,
            'epochs': 1,

            'model_loader': 'Conv1dVAE([128, 10], 3)',
            'optim_loader': 'opt.AdamW(m.parameters(), weight_decay=params["wd"])',
            'wd': 1e-6,

            'k_mse': 1,
            'k_kl': None,

            'console': True,
            'save_rate': 0.0
        })
        print(stdout)


if __name__ == '__main__':
    unittest.main()
