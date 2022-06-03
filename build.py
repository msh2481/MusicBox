import os
import torch
from torch import nn, optim as opt
from torch.optim import lr_scheduler as sch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neptune.new as neptune
from random import random 

def ensure_download(remote_name, local_name=None):
    if local_name not in os.listdir():
        project = neptune.init_project(name="mlxa/MusicBox", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==")
        project[remote_name].download(local_name)
    assert local_name in os.listdir()

def dataset(name):
    if name == 'mnist':
        return datasets.MNIST(root='mnist', train=True, transform=transforms.ToTensor(), download=True)
    elif name == 'dataset_v2':
        ensure_download('dataset_v2', 'dataset_v2.p')
        return torch.load('dataset_v2.p')
    else:
        assert False, f'unknown dataset {name}'

def dataloader(*, data=None, batch_size=None, shuffle=True, num_workers=2, **ignore):
    return DataLoader(dataset(data), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,  pin_memory=False)

def model_optim_sched(*, device=None, model_loader=None, optim_loader=None, sched_loader='sch.ExponentialLR(o, 1.0)', **params):
    from models import ConvDummy, Conv1dAE, Conv1dVAE
    m = eval(model_loader).to(device)
    o = eval(optim_loader)
    s = eval(sched_loader)
    return m, o, s

def kl_div(mu, logsigma):
    assert mu.shape == logsigma.shape
    loss = -0.5 * (1 + logsigma - torch.exp(logsigma) - mu**2).sum(dim=1).mean(dim=0)
    return loss

def criterion(*, k_mse=None, k_kl=None, **ignore):
    if k_kl is None:
        assert k_mse is not None
        def result(input, target, aux):
            mse = F.mse_loss(input, target)
            return k_mse * mse, {'mse': mse}
        return result
    else:
        def result(input, target, aux):
            mse = F.mse_loss(input, target)
            kl = kl_div(input, aux)
            return k_mse * mse + k_kl * kl, {'mse': mse, 'kl': kl}
        return result

def logger(*, console=None, save_rate=None, **ignore):
    if console:
        def result(*, epoch=None, last_batch=None, losses=None, model=None):
            print(f'Epoch: {epoch:.2f}', end='\t')
            for part, value in losses.items():
                print(f'{part}: {value}', end=' ')
            if random() < save_rate or last_batch:
                filename = f'model_{epoch}'
                torch.save(model.state_dict(), filename + '.p')
        return result
    
    run = neptune.init(
        project="mlxa/MusicBox",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
    )
    run['params'] = dict(**ignore, save_rate=None)
    def result(*, epoch=None, last_batch=None, losses=None, model=None):
        for part, value in losses.items():
           run[part].log(value)
        if random() < save_rate or last_batch:
            filename = f'model_{epoch}'
            torch.save(model.state_dict(), filename + '.p')
            run[filename].upload(filename + '.p')
    return result

def run(cfg):
    from train import trainVAE
    trainer = eval(cfg['trainer'])
    m, o, s = model_optim_sched(**cfg)
    trainer(loader=dataloader(**cfg), model=m, optim=o, sched=s, criterion=criterion(**cfg), logger=logger(**cfg), **cfg)