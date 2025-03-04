#include <math.h>
#include <stdlib.h>
#include <string.h>

const char PADDLE = 1, BALL = 2;
const unsigned char NOOP = 0, UP = 1, DOWN = 2;

typedef struct {
    char *obs0, *obs1;
    unsigned char *act0, *act1;
    float *reward0, *reward1, *done0, *done1;
    int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, x, dx;
    float y, dy, max_dy;
} Pong;

void set_obs(Pong* env, char paddle, char ball) {
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

void c_reset(Pong* env) {
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

void c_step(Pong* env) {
    env->reward0[0] = env->reward1[0] = 0;
    env->done0[0] = env->done1[0] = 0;
    set_obs(env, 0, 0);
    if (env->act0[0] == UP && env->paddle0_y > 0) env->paddle0_y--;
    if (env->act0[0] == DOWN && env->paddle0_y < env->size_y - 2) env->paddle0_y++;
    if (env->act1[0] == UP && env->paddle1_y > 0) env->paddle1_y--;
    if (env->act1[0] == DOWN && env->paddle1_y < env->size_y - 2) env->paddle1_y++;
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
        env->reward1[0] = 2 * (float)(env->x == 0) - 1.f;
        env->reward0[0] = -env->reward1[0];
        env->done0[0] = env->done1[0] = 1.f;
        c_reset(env);
    }
    set_obs(env, PADDLE, BALL);
    env->t++;
}
