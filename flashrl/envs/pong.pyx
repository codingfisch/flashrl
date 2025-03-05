# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from 'pong.h':
    ctypedef struct CPong:
        char *obs0
        char *obs1
        unsigned char *act0
        unsigned char *act1
        float *reward0
        float *reward1
        float *done0
        float *done1
        int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, x, dx
        float y, dy, max_dy

    void c_reset(CPong* env)
    void c_step(CPong* env)

cdef class Pong:
    cdef:
        CPong* envs
        int n_agents, n_acts
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef float[:] rewards_memview
        cdef float[:] dones_memview
        int size_x, size_y
        float max_dy

    def __init__(self, n_agents=1, n_acts=3, size_x=16, size_y=8, max_dy=1.):
        self.envs = <CPong*>calloc(n_agents // 2, sizeof(CPong))
        self.n_agents = n_agents
        self.n_acts = n_acts
        self.obs_arr = np.zeros((n_agents, size_y, size_x), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.float32)
        self.dones_arr = np.zeros(n_agents, dtype=np.float32)
        self.obs_memview = self.obs_arr
        self.acts_memview = self.acts_arr
        self.rewards_memview = self.rewards_arr
        self.dones_memview = self.dones_arr
        cdef int i
        for i in range(n_agents // 2):
            env = &self.envs[i]
            env.obs0, env.obs1 = &self.obs_memview[2 * i, 0, 0], &self.obs_memview[2 * i + 1, 0, 0]
            env.act0, env.act1 = &self.acts_memview[2 * i], &self.acts_memview[2 * i + 1]
            env.reward0, env.reward1 = &self.rewards_memview[2 * i], &self.rewards_memview[2 * i + 1]
            env.done0, env.done1 = &self.dones_memview[2 * i], &self.dones_memview[2 * i + 1]
            env.size_x = size_x
            env.size_y = size_y
            env.max_dy = max_dy

    def reset(self, seed=None):
        if seed is not None:
            srand(seed)
        cdef int i
        for i in range(0, self.n_agents // 2):
            c_reset(&self.envs[i])
        return self

    def step(self, np.ndarray acts):
        self.acts_arr[:] = acts[:]
        cdef int i
        for i in range(0, self.n_agents // 2):
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
