# pylint: disable=arguments-differ
import math
import os
from time import time

import neptune.new as neptune
import pytorch_lightning as pl
import soundfile
import torch

from models import MixtureNet, QueueNet
from mu_law import mu_decode
from sampling import beam_search


def is_power(x, a):
    if x < 1:
        return False
    p = math.floor(math.log(x, a))
    assert a**p <= x
    return x - 1 < a**p


class DaNet(pl.LightningModule):
    def __init__(self, model_loader, noise=0.0, lr=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = eval(model_loader)
        self.lr = lr
        self.output_size = 2**12

    def forward(self, x):
        return self.model(x)


model_name = input("Model name: ")
beam_size = int(input("Beam size (default 1): ") or 1)
branch_factor = int(input("Branch factor (default 1): ") or 1)
temperature = float(input("Temperature (default 0.5): ") or 0.5)
run_name = input("Run name (current time by default): ") or hex(int(time()))[2:]


def logging(one_hot, nll):
    alphabet, length = one_hot.size()
    assert alphabet == 256
    if not is_power(length, 1.2):
        return
    mu_encoded = (one_hot.argmax(dim=0) + 0.5) / alphabet - 0.5
    data = mu_decode(mu_encoded).numpy()
    print(f"Writing {run_name}_{length}.wav", flush=True)
    soundfile.write(f"{run_name}_{length}.wav", data, samplerate=22050)


project = neptune.init_project(
    name="mlxa/MusicBox",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTIzY2UxZC1jMjI5LTRlYTQtYjQ0Yi1kM2JhMGU1NDllYTIifQ==",
)
filename = f"{model_name}.ckpt"
if not os.path.exists(filename):
    project[model_name].download(filename)
print("Checkpoint loaded", flush=True)
model = DaNet.load_from_checkpoint(f"{model_name}.ckpt")
print("Model ready", flush=True)
seed = torch.zeros(256, 1)
beam_search(model, beam_size, branch_factor, 2**30, temperature, seed, logging)
