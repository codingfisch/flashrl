# cython: language_level=3
from libc.stdlib cimport free, srand, calloc

cdef extern from 'grid.h':
    ctypedef struct Grid:
        char* obs
        unsigned char* act
        float* reward
        float* done
        int size, t, x, y, goal_x, goal_y

    void c_reset(Grid* env)
    void c_step(Grid* env)


cdef class CyGrid:
    cdef:
        Grid* envs
        size_t n

    def __init__(self, char[:, :, :] obs, unsigned char[:] acts, float[:] rewards, float[:] dones, size_t n, int size):
        self.envs = <Grid*> calloc(n, sizeof(Grid))
        self.n = n
        cdef int i
        for i in range(n):
            self.envs[i] = Grid(&obs[i, 0, 0], &acts[i], &rewards[i], &dones[i], size=size, t=0, x=0, y=0, goal_x=0, goal_y=0)

    def reset(self, seed=None):
        if seed is not None:
            srand(seed)
        cdef int i
        for i in range(self.n):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(self.n):
            c_step(&self.envs[i])

    def close(self):
        free(self.envs)
