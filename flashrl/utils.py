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


def print_curves(data_dict, keys=None, height=8, width=65):
    keys = data_dict.keys() if keys is None else (keys,) if isinstance(keys, str) else keys
    for k in keys:
        fig = plotille.Figure()
        fig.y_label = k
        fig._height = height
        fig._width = width
        fig.scatter(list(range(len(data_dict[k]))), data_dict[k])
        print('\n'.join(fig.show().split('\n')[:-2]))


def render_ascii(obs, obs_max, obs_min=0, fps=4, data=None):
    obs = (23 * (obs - obs_min) / (obs_max - obs_min)).byte().cpu().numpy() + 232
    for i, o in enumerate(obs):#.transpose(0, 2, 1)):
        print(f'step {i}')
        for row in range(o.shape[0]):
            for col in range(o.shape[1]):
                print(f"\033[48;5;{o[row, col]}m  \033[0m", end='')
            if data is not None:
                if row < len(data):
                    print(f'{list(data)[row]}: {list(data.values())[row][i]}', end='')
            print()
        if i < len(obs) - 1:
            time.sleep(1 / fps)
            print(f'\033[A\033[{len(o) + 1}A')


def render_gif(filepath, obs, obs_max, obs_min=0, upscale=64, fps=2, loop=0, data=None):
    font_size = obs.shape[-1] * upscale // 24
    obs = (255 * (obs - obs_min) / (obs_max - obs_min)).byte().cpu().numpy()
    frames = []
    for i, o in enumerate(obs):
        im = Image.fromarray(o).resize((upscale*o.shape[-1], upscale*o.shape[-2]), resample=0)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), f'step: {i}', fill=255, font_size=font_size)
        if data is not None:
            text = []
            for k, v in data.items():
                text.append(f'{k}: {v[i]}')
            draw.text((0, im.size[1] - len(text) * font_size), '\n'.join(text), fill=255, font_size=font_size)
        frames.append(im)
    frames[0].save(filepath, append_images=frames[1:], save_all=True, duration=1000/fps, loop=loop)


def print_table(data, fmt='%.2f'):
    x = np.stack(list(data.values()), 1)
    return np.savetxt(fname=sys.stdout.buffer, X=x, fmt=fmt, delimiter='\t', header='\t'.join(data), comments='')
