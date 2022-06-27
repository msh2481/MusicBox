import os
from random import random

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import soundfile
import torch
from torch import nn
from torch import optim as opt
from torch.nn import functional as F
from torch.optim import lr_scheduler as sch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import *
from train import generate, trainAR


def ensure_download(remote_name, local_name=None):
    if local_name not in os.listdir():
        project = neptune.init_project(
            name="mlxa/MusicBox",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
        )
        project[remote_name].download(local_name)
    assert local_name in os.listdir()


def normalize(x):
    return (x - x.mean()) / x.std()

class OneHotData(torch.utils.data.Dataset):
    def __init__(self, X, y, num_classes, sample_length):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.sample_length = sample_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        indices = self.X[idx]
        assert len(indices.shape) == 1
        start_pos = torch.randint(0, indices.size(0) - self.sample_length + 1, (1,))
        end_pos = start_pos + self.sample_length
        indices = indices[start_pos:end_pos]
        assert indices.shape == (self.sample_length,)
        return F.one_hot(indices.long(), num_classes=self.num_classes).t().float(), self.y[idx]


def dataset_chooser(name):
    if name == "mnist":
        return datasets.MNIST(
            root="mnist",
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(32)]
            ),
            download=True,
        )
    if name == "dataset_v2":
        ensure_download("dataset_v2", "dataset_v2.p")
        X = torch.load("dataset_v2.p")
        return [(X[i], 0) for i in range(len(X))]
    if name == "dataset_v3":
        ensure_download("X_v3", "X_v3.p")
        ensure_download("y_v3", "y_v3.p")
        X, y = torch.load("X_v3.p"), torch.load("y_v3.p")
        return [(X[i].unsqueeze(0), y[i]) for i in range(len(X))]
    if name == "dataset_v4":
        ensure_download("X_v4", "X_v4.p")
        ensure_download("y_v4", "y_v4.p")
        X, y = torch.load("X_v4.p"), torch.load("y_v4.p")
        return [(normalize(X[i]).unsqueeze(0), y[i]) for i in range(len(X))]
    if name == "dataset_v5":
        ensure_download("X_v5", "X_v5.p")
        ensure_download("y_v5", "y_v5.p")
        X, y = torch.load("X_v5.p"), torch.load("y_v5.p")
        return [(X[i], y[i]) for i in range(len(X))]
    if name == "dataset_v6":
        ensure_download("X_v6", "X_v6.p")
        ensure_download("y_v6", "y_v6.p")
        X, y = torch.load("X_v6.p")[:, ::2], torch.load("y_v6.p")
        return OneHotData(X, y, 256, 4096)
    assert False, f"unknown dataset {name}"


def dataset(name, part=1):
    data = dataset_chooser(name)
    full = len(data)
    need = round(full * part)
    return Subset(data, np.arange(need))


def dataloader(
    *,
    data=None,
    part=1,
    batch_size=None,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    **ignore,
):
    return DataLoader(
        dataset(data, part=part),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def model_optim_sched(
    *,
    device=None,
    model_loader=None,
    optim_loader=None,
    sched_loader="sch.ExponentialLR(o, 1.0)",
    **params,
):
    m = eval(model_loader).to(device)
    o = eval(optim_loader)
    s = eval(sched_loader)
    return m, o, s


def criterion(**ignore):
    def result(input, target):
        mse = F.mse_loss(input, target)
        return mse, {"mse": mse}

    return result


class BaseLogger:
    def __init__(self, save_rate=None, sample_rate=None, **ignore):
        self.save_rate = save_rate
        self.sample_rate = sample_rate

    def __call__(self, epoch=None, last_batch=None, losses=None, model=None):
        self.show_losses(epoch, losses)
        if self.save_rate is not None and random() < self.save_rate:
            name = f"model_{epoch}"
            torch.save(model.state_dict(), name + ".p")
            self.upload(name, name + ".p")
        if self.sample_rate is not None and random() < self.sample_rate:
            name = f"sample_{epoch}"
            sample = generate(model, 2**16).detach().cpu().flatten()
            soundfile.write(name + ".wav", sample, 22050)
            self.upload(name, name + ".wav")


class ConsoleLogger(BaseLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def show_losses(self, epoch, losses):
        print()
        print(f"Epoch: {epoch:.2f}", end="\t")
        for part, value in losses.items():
            print(f"{part}: {value}", end=" ")
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
        self.run["params"] = dict(**kwargs)

    def show_losses(self, epoch, losses):
        for name, value in losses.items():
            self.run[name].log(value)

    def upload(self, name, filename):
        self.run[name].upload(filename)


def logger(*, console=None, **kwargs):
    return ConsoleLogger(**kwargs) if console else NeptuneLogger(**kwargs)


def run(cfg):
    trainer = eval(cfg["trainer"])
    m, o, s = model_optim_sched(**cfg)
    return trainer(
        loader=dataloader(**cfg),
        model=m,
        optim=o,
        sched=s,
        criterion=criterion(**cfg),
        logger=logger(**cfg),
        **cfg,
    )


def saved_model(run_name, checkpoint, init_with=None, strict=False):
    run = neptune.init(
        project="mlxa/MusicBox",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
        run=run_name,
        mode="read-only",
    )
    filename = checkpoint + ".p"
    if filename not in os.listdir():
        run[checkpoint].download(filename)
    params = run["params"].fetch()
    params["device"] = "cpu"
    print(params)
    model, _, _ = model_optim_sched(**params)
    if init_with is not None:
        model.encode(torch.randn(init_with))
    model.load_state_dict(torch.load(filename, map_location="cpu"), strict=strict)
    return model
