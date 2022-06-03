import torch
from torch import nn
from itertools import cycle
from tqdm import tqdm


def trainVAE(*, device=None, loader=None, model=None, optim=None, sched=None, criterion=None, logger=None, epochs=None, **ignore):
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch_num, batch in enumerate(loader):
            x = batch.to(device)
            z, aux = model.encode(batch)
            y = model.decode(z, aux)
            l, parts = criterion(input=y, target=x, aux=aux)
            optim.zero_grad()
            l.backward()
            optim.step()
            logger(epoch=(epoch + batch_num / len(loader)),
                   last_batch=(batch_num == len(loader) - 1), losses=parts, model=model)
        sched.step()
