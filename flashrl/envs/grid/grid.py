from flashrl.envs.main import Env
from flashrl.envs.grid.cy_grid import CyGrid


class Grid(Env):
    def __init__(self, n_envs=1, size=9, device=None, dtype='bfloat16'):
        super().__init__(n_envs=n_envs, obs_shape=(size, size), obs_max=2, n_acts=5, device=device, dtype=dtype)
        self.c_envs = CyGrid(self.obs, self.acts, self.rewards, self.dones, n=n_envs, size=size)
