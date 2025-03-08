import sys
import time
import torch
import random
import plotille
import numpy as np
from PIL import Image, ImageDraw


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_ascii_curve(array, label=None, height=8, width=65):
    fig = plotille.Figure()
    fig._height, fig._width = height, width
    fig.y_label = fig.y_label if label is None else label
    fig.scatter(list(range(len(array))), array)
    print('\n'.join(fig.show().split('\n')[:-2]))


def render_ascii(learn, keys=None, fps=4, env_idx=0, obs=None):
    keys = learn.scalar_data_keys if keys is None else [keys] if isinstance(keys, str) else keys
    obs = learn._data['obs'] if obs is None else obs
    obs = (obs - obs.min()) / (obs.max() - obs.min())
    obs = (23 * obs[env_idx]).byte().cpu().numpy() + 232
    for i, o in enumerate(obs):
        print(f'step {i}')
        for row in range(o.shape[0]):
            for col in range(o.shape[1]):
                print(f"\033[48;5;{o[row, col]}m  \033[0m", end='')
            if row < len(keys):
                print(f'{keys[row]}: {learn._data[keys[row]][env_idx, i]:.2g}', end='')
            print()
        if i < len(obs) - 1:
            time.sleep(1 / fps)
            print(f'\033[A\033[{len(o) + 1}A')


def render_gif(filepath, learn, keys=None, upscale=64, fps=2, loop=0, env_idx=0, obs=None):
    keys = learn.scalar_data_keys if keys is None else [keys] if isinstance(keys, str) else keys
    obs = learn._data['obs'] if obs is None else obs
    obs = (obs - obs.min()) / (obs.max() - obs.min())
    obs = (255 * obs[env_idx]).byte().cpu().numpy()
    font_size = obs.shape[-1] * upscale // 32
    frames = []
    for i, o in enumerate(obs):
        im = Image.fromarray(o).resize((upscale*o.shape[-1], upscale*o.shape[-2]), resample=0)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), f'step: {i}', fill=255, font_size=font_size)
        text = [f'{k}: {learn._data[k][env_idx, i]:.2g}' for k in keys]
        draw.text((0, im.size[1] - (len(text) + .5) * font_size), '\n'.join(text), fill=255, font_size=font_size)
        frames.append(im)
    frames[0].save(filepath, append_images=frames[1:], save_all=True, duration=1000/fps, loop=loop)


def print_table(learn, keys=None, fmt='%.2f', env_idx=0):
    keys = learn.scalar_data_keys if keys is None else [keys] if isinstance(keys, str) else keys
    x = np.stack([learn._data[k][env_idx].float().cpu().numpy() for k in keys], 1)
    np.savetxt(fname=sys.stdout.buffer, X=x, fmt=fmt, delimiter='\t', header='\t'.join(keys), comments='')
