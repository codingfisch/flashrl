# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from *:
    '''
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char PADDLE = 1, BALL = 2;
const unsigned char NOOP = 0, UP = 1, DOWN = 2;

typedef struct {
    char *obs0, *obs1;
    unsigned char *act0, *act1;
    char *reward0, *reward1, *done0, *done1;
    int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, x, dx;
    float y, dy, max_dy;
} CPong;

void set_obs(CPong* env, char paddle, char ball) {
    for (int i = -1; i < 2; i++) {
        if (env->paddle0_y + i >= 0 && env->paddle0_y + i <= env->size_y - 1) {
            env->obs0[(env->size_x - 1) - env->paddle0_x + (env->paddle0_y + i) * env->size_x] = paddle;
            env->obs1[env->paddle0_x + (env->paddle0_y + i) * env->size_x] = paddle;
        }
        if (env->paddle1_y + i >= 0 && env->paddle1_y + i <= env->size_y - 1) {
            env->obs0[(env->size_x - 1) - env->paddle1_x + (env->paddle1_y + i) * env->size_x] = paddle;
            env->obs1[env->paddle1_x + (env->paddle1_y + i) * env->size_x] = paddle;
        }
    }
    env->obs0[(env->size_x - 1) - env->x + (int)(roundf(env->y)) * env->size_x] = ball;
    env->obs1[env->x + (int)(roundf(env->y)) * env->size_x] = ball;
}

void c_reset(CPong* env) {
    env->t = 0;
    memset(env->obs0, 0, env->size_x * env->size_y);
    memset(env->obs1, 0, env->size_x * env->size_y);
    env->x = env->size_x / 2;
    env->y = rand() % (env->size_y - 1);
    env->dx = (rand() % 2) ? 1 : -1;
    env->dy = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    env->paddle0_x = 0;
    env->paddle1_x = env->size_x - 1;
    env->paddle0_y = env->paddle1_y = env->size_y / 2;
    set_obs(env, PADDLE, BALL);
}

void c_step(CPong* env) {
    env->reward0[0] = env->reward1[0] = 0;
    env->done0[0] = env->done1[0] = 0;
    set_obs(env, 0, 0);
    unsigned char act0 = env->act0[0];
    unsigned char act1 = env->act1[0];
    if (act0 == UP && env->paddle0_y > 0) env->paddle0_y--;
    if (act0 == DOWN && env->paddle0_y < env->size_y - 2) env->paddle0_y++;
    if (act1 == UP && env->paddle1_y > 0) env->paddle1_y--;
    if (act1 == DOWN && env->paddle1_y < env->size_y - 2) env->paddle1_y++;
    env->dy = fminf(fmaxf(env->dy, -env->max_dy), env->max_dy);
    env->x += env->dx;
    env->y += env->dy;
    env->y = fminf(fmaxf(env->y, 0.f), env->size_y - 1.f);
    if (env->y <= 0 || env->y >= env->size_y - 1) env->dy = -env->dy;
    if (env->x == 1 && env->y >= env->paddle0_y - 1 && env->y <= env->paddle0_y + 1) {
        env->dx = -env->dx;
        env->dy += env->y - env->paddle0_y;
    }
    if (env->x == env->size_x - 2 && env->y >= env->paddle1_y - 1 && env->y <= env->paddle1_y + 1) {
        env->dx = -env->dx;
        env->dy += env->y - env->paddle1_y;
    }
    if (env->x == 0 || env->x == env->size_x - 1) {
        env->reward1[0] = 2 * (char)(env->x == 0) - 1;
        env->reward0[0] = -env->reward1[0];
        env->done0[0] = env->done1[0] = 1;
        c_reset(env);
    }
    set_obs(env, PADDLE, BALL);
    env->t++;
}
'''

    ctypedef struct CPong:
        char *obs0
        char *obs1
        unsigned char *act0
        unsigned char *act1
        char *reward0
        char *reward1
        char *done0
        char *done1
        int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, x, dx
        float y, dy, max_dy

    void c_reset(CPong* env)
    void c_step(CPong* env)

cdef class Pong:
    cdef:
        CPong* envs
        int n_agents, _n_acts
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef char[:] rewards_memview
        cdef char[:] dones_memview
        int size_x, size_y
        float max_dy

    def __init__(self, n_agents=1024, n_acts=3, size_x=16, size_y=8, max_dy=1.):
        self.envs = <CPong*>calloc(n_agents // 2, sizeof(CPong))
        self.n_agents = n_agents
        self._n_acts = n_acts
        self.obs_arr = np.zeros((n_agents, size_y, size_x), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.int8)
        self.dones_arr = np.zeros(n_agents, dtype=np.int8)
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
        for i in range(self.n_agents // 2):
            c_reset(&self.envs[i])
        return self

    def step(self, np.ndarray acts):
        self.acts_arr[:] = acts[:]
        cdef int i
        for i in range(self.n_agents // 2):
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
    def n_acts(self): return self._n_acts
