import os
from matplotlib.cbook import ls_mapper
import torch
from torch import nn, optim as opt
from torch.optim import lr_scheduler as sch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neptune.new as neptune
from random import random
import matplotlib.pyplot as plt 
from models import *

def ensure_download(remote_name, local_name=None):
    if local_name not in os.listdir():
        project = neptune.init_project(name="mlxa/MusicBox", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==")
        project[remote_name].download(local_name)
    assert local_name in os.listdir()

def log1px_encode(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def dataset(name):
    def normalize(x):
        return (x - x.mean()) / x.std()
    
    if name == 'mnist':
        return datasets.MNIST(root='mnist', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(32)]), download=True)
    elif name == 'dataset_v2':
        ensure_download('dataset_v2', 'dataset_v2.p')
        X = torch.load('dataset_v2.p')
        return [(X[i], 0) for i in range(len(X))]
    elif name == 'dataset_v2_overfit':
        X = dataset('dataset_v2')[0][0][:, ::100]
        return [(X, 0)]
    elif name == 'dataset_v3':
        ensure_download('X_v3', 'X_v3.p')
        ensure_download('y_v3', 'y_v3.p')
        X, y = torch.load('X_v3.p'), torch.load('y_v3.p')
        return [(X[i].unsqueeze(0), y[i]) for i in range(len(X))]
    elif name == 'dataset_v4':
        ensure_download('X_v4', 'X_v4.p')
        ensure_download('y_v4', 'y_v4.p')
        X, y = torch.load('X_v4.p'), torch.load('y_v4.p')
        return [(normalize(X[i]).unsqueeze(0), y[i]) for i in range(len(X))]
    elif name == 'dataset_v4_overfit':
        return dataset('dataset_v4')[:4]
    else:
        assert False, f'unknown dataset {name}'

def dataloader(*, data=None, batch_size=None, shuffle=True, num_workers=0, pin_memory=False, **ignore):
    if data == 'dataset_v2_overfit':
        return dataset('dataset_v2_overfit')
    return DataLoader(dataset(data), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

def model_optim_sched(*, device=None, model_loader=None, optim_loader=None, sched_loader='sch.ExponentialLR(o, 1.0)', **params):
    m = eval(model_loader).to(device)
    o = eval(optim_loader)
    s = eval(sched_loader)
    return m, o, s

def kl_div(mu, logsigma):
    assert mu.shape == logsigma.shape
    loss = -0.5 * (1 + logsigma - torch.exp(logsigma) - mu**2).sum(dim=1).mean(dim=0)
    return loss

def criterion(**ignore):
    def result(input, target):
        mse = F.mse_loss(input, target)
        return mse, {'mse': mse}
    return result

class BaseLogger:
    def __init__(self, save_rate=None, sample_rate=None, **ignore):
        self.save_rate = save_rate 
        self.sample_rate = sample_rate
    def __call__(self, epoch=None, last_batch=None, losses=None, model=None):
        self.show_losses(epoch, losses)
        if self.save_rate is not None and random() < self.save_rate:
            name = f'model_{epoch}'
            torch.save(model.state_dict(), name + '.p')
            self.upload(name, name + '.p')
        if self.sample_rate is not None and random() < self.sample_rate:
            name = f'sample_{epoch}'
            sample = model.generate().detach().cpu().flatten(start_dim=0, end_dim=-2)
            mu, sigma = sample.mean(), sample.std()
            sample = sample.clip(mu - 3 * sigma, mu + 3 * sigma)
            plt.imshow(sample)
            plt.savefig(name + '.jpg')
            self.upload(name, name + '.jpg')

class ConsoleLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def show_losses(self, epoch, losses):
        print()
        print(f'Epoch: {epoch:.2f}', end='\t')
        for part, value in losses.items():
            print(f'{part}: {value}', end=' ')
        print(flush=True)
    def upload(self, name, filename):
        pass

class NeptuneLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run = neptune.init(
            project="mlxa/MusicBox",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
        )
        self.run['params'] = dict(**kwargs)
    def show_losses(self, epoch, losses):
        for name, value in losses.items():
            self.run[name].log(value)
    def upload(self, name, filename):
        self.run[name].upload(filename)

def logger(*, console=None, **kwargs):
    return ConsoleLogger(**kwargs) if console else NeptuneLogger(**kwargs)

def run(cfg):
    from train import trainVAE
    trainer = eval(cfg['trainer'])
    m, o, s = model_optim_sched(**cfg)
    return trainer(loader=dataloader(**cfg), model=m, optim=o, sched=s, criterion=criterion(**cfg), logger=logger(**cfg), **cfg)

def saved_model(run_name, checkpoint, init_with=None, strict=False):
    run = neptune.init(
        project="mlxa/MusicBox",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
        run=run_name,
        mode='read-only'
    )
    filename = checkpoint + '.p'
    if filename not in os.listdir():
        run[checkpoint].download(filename)
    params = run['params'].fetch()
    params['device'] = 'cpu'
    print(params)
    model, _, _ = model_optim_sched(**params)
    if init_with is not None:
        model.encode(torch.randn(init_with))
    model.load_state_dict(torch.load(filename, map_location='cpu'), strict=strict)
    return model