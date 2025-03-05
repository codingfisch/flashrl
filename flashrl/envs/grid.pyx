# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from 'grid.h':
    ctypedef struct CGrid:
        char *obs
        unsigned char *act
        float *reward
        float *done
        int size, t, x, y, goal_x, goal_y

    void c_reset(CGrid *env)
    void c_step(CGrid *env)


cdef class Grid:
    cdef:
        CGrid *envs
        int n_agents, n_acts
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef float[:] rewards_memview
        cdef float[:] dones_memview
        int size

    def __init__(self, n_agents=1, n_acts=5, size=8):
        self.envs = <CGrid*> calloc(n_agents, sizeof(CGrid))
        self.n_agents = n_agents
        self.n_acts = n_acts
        self.obs_arr = np.zeros((n_agents, size, size), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.float32)
        self.dones_arr = np.zeros(n_agents, dtype=np.float32)
        self.obs_memview = self.obs_arr
        self.acts_memview = self.acts_arr
        self.rewards_memview = self.rewards_arr
        self.dones_memview = self.dones_arr
        cdef int i
        for i in range(n_agents):
            env = &self.envs[i]
            env.obs = &self.obs_memview[i, 0, 0]
            env.act = &self.acts_memview[i]
            env.reward = &self.rewards_memview[i]
            env.done = &self.dones_memview[i]
            env.size = size

    def reset(self, seed=None):
        if seed is not None:
            srand(seed)
        cdef int i
        for i in range(self.n_agents):
            c_reset(&self.envs[i])
        return self

    def step(self, np.ndarray acts):
        self.acts_arr[:] = acts[:]
        cdef int i
        for i in range(self.n_agents):
            c_step(&self.envs[i])

    def close(self):
        free(self.envs)

    @property
    def obs(self): return self.obs_arr

    @property
    def acts(self): return self.acts_arr

    @property
    def rewards(self): return self.rewards_arr

    @property
    def dones(self): return self.dones_arr

    @property
    def n_acts(self): return self.n_acts
