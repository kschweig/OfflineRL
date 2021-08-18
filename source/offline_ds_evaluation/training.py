import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _forward(network: nn.Module, data: DataLoader, metric: callable):

    for x, y in data:

        if isinstance(x, list):
            x = [x_.to(next(network.parameters()).device) for x_ in x]
        else:
            x = x.to(next(network.parameters()).device)

        y_hat = network.forward(x).cpu()

        batch_size = y_hat.shape[0]

        if len(y.view(-1)) > batch_size:
            y = y.reshape(batch_size, -1)
        else:
            y = y.reshape(-1)

        loss = metric(y_hat, y)
        yield loss


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: torch.optim.Optimizer) -> list:

    network.train()

    errs = []
    for err in _forward(network, data, loss):
        errs.append(err.item())
        opt.zero_grad()
        try:
            err.backward()
            opt.step()
        except:
            print('error in update step')
    return errs


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> float:

    network.eval()

    performance = []
    for p in _forward(network, data, metric):
        performance.append(p.item())
    return np.mean(performance).item()