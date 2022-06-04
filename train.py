import torch
from torch import nn
from itertools import cycle
from tqdm import tqdm


def trainVAE(*, device=None, loader=None, model=None, optim=None, sched=None, criterion=None, logger=None, epochs=None, **ignore):
    model.train()
    for epoch in tqdm(range(epochs)):
        ls = []
        for batch_num, x in enumerate(loader):
            x = x.to(device)
            z, aux = model.encode(x)
            y = model.decode(z, aux)
            l, parts = criterion(input=y, target=x, aux=aux)
            optim.zero_grad()
            l.backward()
            optim.step()
            ls.append(l.item())
            logger(epoch=(epoch + batch_num / len(loader)),
                   last_batch=(batch_num == len(loader) - 1), losses=parts, model=model)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(sum(ls) / len(ls))
        else:
            sched.step()
