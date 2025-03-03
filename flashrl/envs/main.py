import time
import torch
import numpy as np
from PIL import Image, ImageDraw


class Env:
    def __init__(self, n_envs, obs_shape, n_acts, obs_max, obs_min=0, device=None, dtype='bfloat16'):
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.n_acts = n_acts
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.obs = np.zeros((n_envs, *obs_shape), dtype=np.int8)
        self.acts = np.zeros(n_envs, dtype=np.uint8)
        self.rewards = np.zeros(n_envs, dtype=np.float32)
        self.dones = np.zeros(n_envs, dtype=np.float32)
        self.c_envs = None

    def reset(self, seed=None):
        self.c_envs.reset(seed)
        return self

    def step(self, acts):
        self.acts[:] = acts
        self.c_envs.step()

    def close(self):
        self.c_envs.close()

    def get_tensor(self, x, device=None, dtype=None, non_blocking=True):
        device = device or self.device
        dtype = dtype or self.dtype
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=non_blocking)


def render_ascii(obs, obs_max, obs_min=0, fps=4, data=None):
    obs = (23 * (obs.astype(np.float32) - obs_min) / (obs_max - obs_min)).astype(np.uint8) + 232
    for i, o in enumerate(obs):
        print(f'step {i}')
        for row in range(o.shape[0]):
            for col in range(o.shape[1]):
                print(f"\033[48;5;{o[row, col]}m  \033[0m", end='')
            if row < len(data):
                print(f'{list(data)[row]}: {list(data.values())[row][i]}', end='')
            print()
        if i < len(obs) - 1:
            time.sleep(1 / fps)
            print(f'\033[A\033[{len(o) + 1}A')


def render_gif(filepath, obs, obs_max, obs_min=0, upscale=64, fps=2, loop=0, data=None):
    font_size = obs.shape[-1] * upscale // 24
    obs = (255 * (obs.astype(np.float32) - obs_min) / (obs_max - obs_min)).astype(np.uint8)
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
