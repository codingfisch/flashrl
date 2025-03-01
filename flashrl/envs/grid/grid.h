#include <stdlib.h>
#include <string.h>

const char AGENT = 1, GOAL = 2;
const unsigned char NOOP = 0, LEFT = 1, RIGHT = 2, UP = 3, DOWN = 4;

typedef struct {
    char* obs;
    unsigned char* act;
    float* reward;
    float* done;
    int size, t, x, y, goal_x, goal_y;
} Grid;

void c_reset(Grid* env) {
    env->t = 0;
    memset(env->obs, 0, env->size * env->size);
    env->x = env->y = env->size / 2;
    env->obs[env->x + env->y * env->size] = AGENT;
    env->goal_x = rand() % env->size;
    env->goal_y = rand() % env->size;
    if (env->goal_x == env->x && env->goal_y == env->y) env->goal_x++;
    env->obs[env->goal_x + env->goal_y * env->size] = GOAL;
}

void c_step(Grid* env) {
    env->reward[0] = 0;
    env->done[0] = 0;
    env->obs[env->x + env->y * env->size] = 0;
    if (env->act[0] == LEFT) env->x--;
    else if (env->act[0] == RIGHT) env->x++;
    else if (env->act[0] == UP) env->y--;
    else if (env->act[0] == DOWN) env->y++;
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
