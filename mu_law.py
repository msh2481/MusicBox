import torch

def mu_encode(x):
    mu = torch.tensor(255.)
    return torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / torch.log(1 + mu)

def mu_decode(x):
    mu = torch.tensor(255.)
    return torch.sign(x) * (torch.exp(torch.log(1 + mu) * torch.abs(x)) - 1) / mu