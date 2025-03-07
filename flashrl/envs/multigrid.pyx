# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from *:
    '''
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

const char WALL = 1, AGENT = 2, GOAL = 3;
const unsigned char NOOP = 0, DOWN = 1, UP = 2, LEFT = 3, RIGHT = 4;

typedef struct {
    char *obs;
    unsigned char *acts;
    char *rewards, *dones;
    unsigned char *x, *y;
    char *total_obs;
    int n_agents_per_env, vision, size, t, goal_x, goal_y;
} CMultiGrid;

void get_obs(CMultiGrid* env) {
    int ob_size = 2 * env->vision + 1;
    int ob_pixels = ob_size * ob_size;
    memset(env->obs, 0, env->n_agents_per_env * ob_pixels);
    int center = ob_pixels / 2;
    for (int i = 0; i < env->n_agents_per_env; i++) {
        for (int x = -env->vision; x <= env->vision; x++) {
            for (int y = -env->vision; y <= env->vision; y++) {
                char world_x = env->x[i] + x;
                char world_y = env->y[i] + y;
                if (world_x < 0 || world_x > env->size - 1 || world_y < 0 || world_y > env->size - 1)  {
                    env->obs[i * ob_pixels + center + y * ob_size + x] = WALL;
                }
            }
        }
        for (int j = 0; j < env->n_agents_per_env; j++) {
            char dx = env->x[j] - env->x[i];
            char dy = env->y[j] - env->y[i];
            if (abs(dx) <= env->vision && abs(dy) <= env->vision) {
                env->obs[i * ob_pixels + center + dy * ob_size + dx] = AGENT;
            }
        }
        char dx = env->goal_x - env->x[i];
        char dy = env->goal_y - env->y[i];
        if (abs(dx) <= env->vision && abs(dy) <= env->vision) {
            env->obs[i * ob_pixels + center + dy * ob_size + dx] = GOAL;
        }
    }
}

void get_total_obs(CMultiGrid* env) {
    memset(env->total_obs, 0, env->size * env->size);
    for (int i = 0; i < env->n_agents_per_env; i++) {
        env->total_obs[env->x[i] + env->y[i] * env->size] = AGENT;
    }
    env->total_obs[env->goal_x + env->goal_y * env->size] = GOAL;
}

void c_reset(CMultiGrid* env, bool with_total_obs) {
    env->t = 0;
    env->goal_x = rand() % env->size;
    env->goal_y = rand() % env->size;
    for (int i = 0; i < env->n_agents_per_env; i++) {
        env->x[i] = rand() % env->size;
        env->y[i] = rand() % env->size;
    }
    get_obs(env);
    if (with_total_obs) {
        get_total_obs(env);
    }
}

void agent_step(CMultiGrid* env, int i, bool with_total_obs) {
    env->rewards[i] = 0;
    env->dones[i] = 0;
    unsigned char act = env->acts[i];
    if (act == LEFT) env->x[i]--;
    else if (act == RIGHT) env->x[i]++;
    else if (act == UP) env->y[i]--;
    else if (act == DOWN) env->y[i]++;
    if (env->t > 3 * env->size || env->x[i] < 0 || env->y[i] < 0 || env->x[i] >= env->size || env->y[i] >= env->size) {
        env->dones[i] = 1;
        env->rewards[i] = -1;
        c_reset(env, with_total_obs);
        return;
    }
    if (env->x[i] == env->goal_x && env->y[i] == env->goal_y) {
        env->dones[i] = 1;
        env->rewards[i] = 1;
        c_reset(env, with_total_obs);
        return;
    }
}

void c_step(CMultiGrid* env, bool with_total_obs) {
    for (int i = 0; i < env->n_agents_per_env; i++){
        agent_step(env, i, with_total_obs);
    }
    get_obs(env);
    if (with_total_obs) {
        get_total_obs(env);
    }
    env->t++;
}
'''

    ctypedef struct CMultiGrid:
        char *obs
        unsigned char *acts
        char *rewards
        char *dones
        unsigned char *x
        unsigned char *y
        char *total_obs
        int n_agents_per_env, vision, size, t, goal_x, goal_y

    void c_reset(CMultiGrid* env, bint with_total_obs)
    void c_step(CMultiGrid* env, bint with_total_obs)

cdef class MultiGrid:
    cdef:
        CMultiGrid* envs
        int n_agents, _n_acts, _n_agents_per_env
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr, x_arr, y_arr, total_obs_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef char[:] rewards_memview
        cdef char[:] dones_memview
        cdef unsigned char[:] x_memview
        cdef unsigned char[:] y_memview
        cdef char[:, :, :] total_obs_memview
        int size
        bint with_total_obs

    def __init__(self, n_agents=1024, n_acts=5, n_agents_per_env=2, vision=3, size=8):
        self.envs = <CMultiGrid*>calloc(n_agents // n_agents_per_env, sizeof(CMultiGrid))
        self.n_agents = n_agents
        self._n_acts = n_acts
        self._n_agents_per_env = n_agents_per_env
        self.obs_arr = np.zeros((n_agents, 2*vision+1, 2*vision+1), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.int8)
        self.dones_arr = np.zeros(n_agents, dtype=np.int8)
        self.x_arr = np.zeros(n_agents, dtype=np.uint8)
        self.y_arr = np.zeros(n_agents, dtype=np.uint8)
        self.total_obs_arr = np.zeros((n_agents // n_agents_per_env, size, size), dtype=np.int8)
        self.obs_memview = self.obs_arr
        self.acts_memview = self.acts_arr
        self.rewards_memview = self.rewards_arr
        self.dones_memview = self.dones_arr
        self.x_memview = self.x_arr
        self.y_memview = self.y_arr
        self.total_obs_memview = self.total_obs_arr
        cdef int i, i_agent
        for i in range(n_agents // n_agents_per_env):
            env = &self.envs[i]
            i_agent = n_agents_per_env * i
            env.obs = &self.obs_memview[i_agent, 0, 0]
            env.acts = &self.acts_memview[i_agent]
            env.rewards = &self.rewards_memview[i_agent]
            env.dones = &self.dones_memview[i_agent]
            env.x = &self.x_memview[i_agent]
            env.y = &self.y_memview[i_agent]
            env.total_obs = &self.total_obs_memview[i, 0, 0]
            env.n_agents_per_env = n_agents_per_env
            env.vision = vision
            env.size = size

    def reset(self, seed=None, with_total_obs=False):
        if seed is not None:
            srand(seed)
        cdef int i
        #cdef bool c_with_total_obs = with_total_obs
        for i in range(self.n_agents // self.n_agents_per_env):
            c_reset(&self.envs[i], with_total_obs)
        return self

    def step(self, np.ndarray acts, with_total_obs=False):
        self.acts_arr[:] = acts[:]
        cdef int i
        #cdef bool c_with_total_obs = with_total_obs
        for i in range(self.n_agents // self.n_agents_per_env):
            c_step(&self.envs[i], with_total_obs)

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

    @property
    def n_agents_per_env(self): return self._n_agents_per_env

    @property
    def total_obs(self): return self.total_obs_arr
