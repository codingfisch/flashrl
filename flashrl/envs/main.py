import torch
import numpy as np


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
