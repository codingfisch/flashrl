# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from *:
    '''
#include <stdlib.h>
#include <string.h>

const char AGENT = 1, GOAL = 2;
const unsigned char NOOP = 0, LEFT = 1, RIGHT = 2, UP = 3, DOWN = 4;

typedef struct {
    char *obs;
    unsigned char *act;
    char *reward, *done;
    int size, t, x, y, goal_x, goal_y;
} CGrid;

void c_reset(CGrid* env) {
    env->t = 0;
    memset(env->obs, 0, env->size * env->size);
    env->x = env->y = env->size / 2;
    env->obs[env->x + env->y * env->size] = AGENT;
    env->goal_x = rand() % env->size;
    env->goal_y = rand() % env->size;
    if (env->goal_x == env->x && env->goal_y == env->y) env->goal_x++;
    env->obs[env->goal_x + env->goal_y * env->size] = GOAL;
}

void c_step(CGrid* env) {
    env->reward[0] = 0;
    env->done[0] = 0;
    env->obs[env->x + env->y * env->size] = 0;
    unsigned char act = env->act[0];
    if (act == LEFT) env->x--;
    else if (act == RIGHT) env->x++;
    else if (act == UP) env->y--;
    else if (act == DOWN) env->y++;
    if (env->t > 3 * env->size || env->x < 0 || env->y < 0 || env->x >= env->size || env->y >= env->size) {
        env->reward[0] = -1;
        env->done[0] = 1;
        c_reset(env);
        return;
    }
    int position = env->x + env->y * env->size;
    if (env->obs[position] == GOAL) {
        env->reward[0] = 1;
        env->done[0] = 1;
        c_reset(env);
        return;
    }
    env->obs[position] = AGENT;
    env->t++;
}
'''

    ctypedef struct CGrid:
        char *obs
        unsigned char *act
        char *reward
        char *done
        int size, t, x, y, goal_x, goal_y

    void c_reset(CGrid *env)
    void c_step(CGrid *env)


cdef class Grid:
    cdef:
        CGrid *envs
        int n_agents, _n_acts
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef char[:] rewards_memview
        cdef char[:] dones_memview
        int size

    def __init__(self, n_agents=1024, n_acts=5, size=8):
        self.envs = <CGrid*> calloc(n_agents, sizeof(CGrid))
        self.n_agents = n_agents
        self._n_acts = n_acts
        self.obs_arr = np.zeros((n_agents, size, size), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.int8)
        self.dones_arr = np.zeros(n_agents, dtype=np.int8)
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
    def n_acts(self): return self._n_acts
