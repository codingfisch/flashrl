#include <stdlib.h>
#include <string.h>

const char PADDLE = 1, BALL = 2;
const unsigned char NOOP = 0, UP = 1, DOWN = 2;

typedef struct {
    char *obs0, *obs1;
    unsigned char *act0, *act1;
    float *reward0, *reward1, *done0, *done1;
    int size_x, size_y, t, paddle0_x, paddle0_y, paddle1_x, paddle1_y, ball_x, ball_y, ball_dx, ball_dy;
} Pong;

void c_reset(Pong* env) {
    env->t = 0;
    memset(env->obs0, 0, env->size_x * env->size_y);
    memset(env->obs1, 0, env->size_x * env->size_y);
    env->ball_x = env->size_x / 2;
    env->ball_y = env->size_y / 2;
    env->ball_dx = (rand() % 2) ? 1 : -1;
    env->ball_dy = (rand() % 2) ? 1 : -1;
    env->paddle0_x = 0;
    env->paddle1_x = env->size_x - 1;
    env->paddle0_y = env->paddle1_y = env->size_y / 2;
    env->reward0[0] = env->reward1[0] = 0;
    env->done0[0] = env->done1[0] = 0;
}

void c_step(Pong* env) {
    env->obs0[(env->size_x - 1) - env->paddle0_x + env->paddle0_y * env->size_x] = 0;
    env->obs0[(env->size_x - 1) - env->paddle1_x + env->paddle1_y * env->size_x] = 0;
    env->obs1[env->paddle0_x + env->paddle0_y * env->size_x] = 0;
    env->obs1[env->paddle1_x + env->paddle1_y * env->size_x] = 0;
    env->obs0[(env->size_x - 1) - env->ball_x + env->ball_y * env->size_x] = 0;
    env->obs1[env->ball_x + env->ball_y * env->size_x] = 0;
    if (env->act0[0] == UP && env->paddle0_y > 0) env->paddle0_y--;
    if (env->act0[0] == DOWN && env->paddle0_y < env->size_y - 2) env->paddle0_y++;
    if (env->act1[0] == UP && env->paddle1_y > 0) env->paddle1_y--;
    if (env->act1[0] == DOWN && env->paddle1_y < env->size_y - 2) env->paddle1_y++;
    env->ball_x += env->ball_dx;
    env->ball_y += env->ball_dy;
    if (env->ball_y <= 0 || env->ball_y >= env->size_y - 1) env->ball_dy = -env->ball_dy;
    if (env->ball_x == 1 && env->ball_y >= env->paddle0_y && env->ball_y <= env->paddle0_y + 1)
        env->ball_dx = -env->ball_dx;
    if (env->ball_x == env->size_x - 2 && env->ball_y >= env->paddle1_y && env->ball_y <= env->paddle1_y + 1)
        env->ball_dx = -env->ball_dx;
    if (env->ball_x <= 0) { env->reward0[0]--; env->reward1[0]++; env->done0[0]++; env->done1[0]++; c_reset(env); }
    if (env->ball_x >= env->size_x - 1) { env->reward0[0]++; env->reward1[0]--;  env->done0[0]++; env->done1[0]++; c_reset(env); }
    env->obs0[(env->size_x - 1) - env->paddle0_x + env->paddle0_y * env->size_x] = PADDLE;
    env->obs0[(env->size_x - 1) - env->paddle1_x + env->paddle1_y * env->size_x] = PADDLE;
    env->obs1[env->paddle0_x + env->paddle0_y * env->size_x] = PADDLE;
    env->obs1[env->paddle1_x + env->paddle1_y * env->size_x] = PADDLE;
    env->obs0[(env->size_x - 1) - env->ball_x + env->ball_y * env->size_x] = BALL;
    env->obs1[env->ball_x + env->ball_y * env->size_x] = BALL;
    env->t++;
}
