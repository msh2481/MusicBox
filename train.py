import torch
from torch import nn
from itertools import cycle
from tqdm import tqdm


def trainVAE(*, device=None, loader=None, model=None, optim=None, sched=None, criterion=None, logger=None, epochs=None, **ignore):
    model.train()
    for epoch in tqdm(range(epochs)):
        ls = []
        for batch_num, batch in enumerate(loader):
            print(flush=True)
            x = batch.to(device)
            z, aux = model.encode(batch)
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
        for param_group in optim.param_groups:
            print(param_group['lr'])
