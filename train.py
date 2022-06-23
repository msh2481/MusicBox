from itertools import cycle

import optuna
import torch
from torch import nn
from tqdm import tqdm


def trainAR(*, trial=None, device=None, loader=None, noise=0.0, model=None, optim=None, sched=None, criterion=None, logger=None, epochs=None, **ignore):
    model.train()
    last_loss = None
    for epoch in tqdm(range(epochs)):
        ls = []
        for batch_num, (x, info) in enumerate(loader):
            x = x.to(device)
            aug = x.clone() + torch.randn_like(x) * noise * x.std()
            p = model(aug)
            l, parts = criterion(input=p, target=x)
            optim.zero_grad()
            l.backward()
            optim.step()
            ls.append(l.item())
            logger(epoch=(epoch + batch_num / len(loader)),
                   last_batch=(batch_num == len(loader) - 1), losses=parts, model=model)
        last_loss = sum(ls) / len(ls)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(last_loss)
        elif sched is not None:
            sched.step()
        if trial:
            trial.report(last_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return last_loss
