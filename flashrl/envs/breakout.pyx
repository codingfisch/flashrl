# cython: language_level=3
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, srand, calloc

cdef extern from *:
    '''
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char PADDLE = 1, BALL = 2, BRICK = 3;
const unsigned char NOOP = 0, LEFT = 1, RIGHT = 2;

typedef struct {
    char *obs;
    unsigned char *act;
    char *reward;
    char *done;
    int size_x, size_y, t, paddle_x, paddle_y, ball_x, ball_y;
    float ball_dx, ball_dy, max_speed;
    char *bricks;  // Array to track brick states
    int brick_rows, brick_cols;
} CBreakout;

void set_obs(CBreakout* env) {
    memset(env->obs, 0, env->size_x * env->size_y);
    for (int i = -1; i <= 1; i++) {
        if (env->paddle_x + i >= 0 && env->paddle_x + i < env->size_x) {
            env->obs[env->paddle_x + i + env->paddle_y * env->size_x] = PADDLE;
        }
    }
    for (int y = 1; y < env->brick_rows + 1; y++) {
        for (int x = 0; x < env->brick_cols; x++) {
            if (env->bricks[y * env->brick_cols + x]) {
                env->obs[x + y * env->size_x] = BRICK;
            }
        }
    }
    env->obs[env->ball_x + env->ball_y * env->size_x] = BALL;
}

void c_reset(CBreakout* env) {
    env->t = 0;
    memset(env->obs, 0, env->size_x * env->size_y);
    env->paddle_x = env->size_x / 2;
    env->paddle_y = env->size_y - 1;
    env->ball_x = env->paddle_x;
    env->ball_y = env->paddle_y - 1;
    env->ball_dx = (float)rand() / RAND_MAX;
    env->ball_dy = -1.f;
    env->brick_cols = env->size_x;
    env->brick_rows = env->size_y / 4;
    memset(env->bricks, 1, env->brick_rows * env->brick_cols);
    set_obs(env);
}

void c_step(CBreakout* env) {
    env->reward[0] = 0;
    env->done[0] = 0;
    unsigned char act = env->act[0];
    if (act == LEFT && env->paddle_x > 1) env->paddle_x--;
    if (act == RIGHT && env->paddle_x < env->size_x - 2) env->paddle_x++;
    float next_x = env->ball_x + env->ball_dx;
    float next_y = env->ball_y + env->ball_dy;
    if (next_x < 0 || next_x + 0.2f >= env->size_x) env->ball_dx = -env->ball_dx;
    if (next_y < 0) env->ball_dy = -env->ball_dy;
    if ((int)(next_y + 1.5f) == env->paddle_y &&
        next_x >= env->paddle_x - 1 &&
        next_x <= env->paddle_x + 1) {
        env->ball_dy = -env->ball_dy;
        env->ball_dx += next_x - env->paddle_x;
    }
    int brick_x = (int)(next_x);
    int brick_y = (int)(next_y);
    if (brick_y < env->brick_rows && brick_x >= 0 && brick_x < env->brick_cols) {
        int brick_idx = brick_y * env->brick_cols + brick_x;
        if (env->bricks[brick_idx]) {
            env->bricks[brick_idx] = 0;
            env->ball_dy = -env->ball_dy;
            env->reward[0] = 1;  // Reward for breaking brick
        }
    }
    env->ball_dx = fminf(fmaxf(env->ball_dx, -env->max_speed), env->max_speed);
    env->ball_dy = fminf(fmaxf(env->ball_dy, -env->max_speed), env->max_speed);
    env->ball_x = (int)(next_x + 0.5f);
    if (env->ball_x > env->size_x - 1) env->ball_x = env->size_x - 1;
    env->ball_y = (int)(next_y + 0.5f);
    if (env->ball_y >= env->size_y) {
        env->reward[0] = -1;
        env->done[0] = 1;
        c_reset(env);
    }
    int bricks_left = 0;
    for (int i = 0; i < env->brick_rows * env->brick_cols; i++) {
        bricks_left += env->bricks[i];
    }
    if (bricks_left == 0) {
        env->reward[0] = 10;  // Bonus for winning
        env->done[0] = 1;
        c_reset(env);
    }
    set_obs(env);
    env->t++;
}
'''

    ctypedef struct CBreakout:
        char *obs
        unsigned char *act
        char *reward
        char *done
        int size_x, size_y, t, paddle_x, paddle_y, ball_x, ball_y
        float ball_dx, ball_dy, max_speed
        char *bricks
        int brick_rows, brick_cols

    void c_reset(CBreakout* env)
    void c_step(CBreakout* env)

cdef class Breakout:
    cdef:
        CBreakout* envs
        int n_agents, _n_acts
        np.ndarray obs_arr, acts_arr, rewards_arr, dones_arr, bricks_arr
        cdef char[:, :, :] obs_memview
        cdef unsigned char[:] acts_memview
        cdef char[:] rewards_memview
        cdef char[:] dones_memview
        cdef char[:, :] bricks_memview
        int size_x, size_y
        float max_speed
        dict _emoji_map

    def __init__(self, n_agents=2**14, n_acts=3, size_x=16, size_y=12, max_speed=1.0, emoji_map={0: '  ', 1: '🔲', 2: '🔴', 3: '🧱'}):
        self.envs = <CBreakout*>calloc(n_agents, sizeof(CBreakout))
        self.n_agents = n_agents
        self._n_acts = n_acts
        self.obs_arr = np.zeros((n_agents, size_y, size_x), dtype=np.int8)
        self.acts_arr = np.zeros(n_agents, dtype=np.uint8)
        self.rewards_arr = np.zeros(n_agents, dtype=np.int8)
        self.dones_arr = np.zeros(n_agents, dtype=np.int8)
        self.bricks_arr = np.ones((n_agents, size_y//4 * size_x), dtype=np.int8)  # Brick states
        self.obs_memview = self.obs_arr
        self.acts_memview = self.acts_arr
        self.rewards_memview = self.rewards_arr
        self.dones_memview = self.dones_arr
        self.bricks_memview = self.bricks_arr
        self.size_x = size_x
        self.size_y = size_y
        self.max_speed = max_speed

        cdef int i
        for i in range(n_agents):
            env = &self.envs[i]
            env.obs = &self.obs_memview[i, 0, 0]
            env.act = &self.acts_memview[i]
            env.reward = &self.rewards_memview[i]
            env.done = &self.dones_memview[i]
            env.bricks = &self.bricks_memview[i, 0]
            env.size_x = size_x
            env.size_y = size_y
            env.max_speed = max_speed
        self._emoji_map = emoji_map

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

    @property
    def emoji_map(self): return self._emoji_map
