import sys
import torch
import random
import plotille
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_plot(x, name, height=8, width=65):
    fig = plotille.Figure()
    fig.y_label = name
    fig._height = height
    fig._width = width
    fig.scatter(list(range(len(x))), x)
    print('\n'.join(fig.show().split('\n')[:-2]))


def print_table(data, fmt='%.2f'):
    x = np.stack(list(data.values()), 1)
    return np.savetxt(fname=sys.stdout.buffer, X=x, fmt=fmt, delimiter='\t', header='\t'.join(data), comments='')
