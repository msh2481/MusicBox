import os
import torch
from torch import nn, optim as opt
from torch.optim import lr_scheduler as sch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import neptune.new as neptune

'''
cfg:
    dataset
    batch_size
    epochs

    model_loader(receive size here)

    optim_loader
    lr
    betas
    ...

    sched_loader
    lambda
    ...

train_vae(cfg)
    build everything and start neptune run
'''

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

def dataloader(name, batch_size, num_workers):
    return DataLoader(dataset(name), batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)

def model_optim_sched(device, model_loader, optim_loader, sched_loader):
    from models import ConvDummy
    m = eval(model_loader).to(device)
    o = eval(optim_loader)
    s = eval(sched_loader)
    return m, o, s