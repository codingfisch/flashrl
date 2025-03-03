# cython: language_level=3
from libc.stdlib cimport free, srand, calloc

cdef extern from 'pong.h':
    ctypedef struct Pong:
        char *obs0
        char *obs1
        unsigned char *act0
        unsigned char *act1
        float *reward0
        float *reward1
        float *done0
        float *done1
        int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, ball_x, ball_y, ball_dx, ball_dy

    void c_reset(Pong* env)
    void c_step(Pong* env)


cdef class CyPong:
    cdef:
        Pong* envs
        size_t n
        int n_agents

    def __init__(self, char[:, :, :] obs, unsigned char[:] acts, float[:] rewards, float[:] dones, size_t n,
                 int size_x, int size_y):
        self.envs = <Pong*>calloc(n, sizeof(Pong))
        self.n = n
        cdef int i
        for i in range(0, self.n, 2):
            self.envs[i//2] = Pong(&obs[i, 0, 0], &obs[i+1, 0, 0], &acts[i], &acts[i+1], &rewards[i], &rewards[i+1],
                                   &dones[i], &dones[i+1], size_x=size_x, size_y=size_y, t=0, paddle0_x=0, paddle0_y=0,
                                   paddle1_x=0, paddle1_y=0, ball_x=0, ball_y=0, ball_dx=0, ball_dy=0)

    def reset(self, seed=None):
        if seed is not None:
            srand(seed)
        cdef int i
        for i in range(0, self.n // 2):
            c_reset(&self.envs[i])

    def step(self):
        cdef int i
        for i in range(0, self.n // 2):
            c_step(&self.envs[i])

    def close(self):
        free(self.envs)
