from flashrl.envs.main import Env
from flashrl.envs.pong.cy_pong import CyPong


class Pong(Env):
    def __init__(self, n_envs=1, size_x=16, size_y=9, device=None, dtype='bfloat16'):
        super().__init__(n_envs=n_envs, obs_shape=(size_x, size_y), obs_max=2, n_acts=3, device=device, dtype=dtype)
        self.c_envs = CyPong(self.obs, self.acts, self.rewards, self.dones, n=n_envs, size_x=size_x, size_y=size_y)
