import argparse
from flashrl import HPARAMS, set_seed, print_table, Learner, LSTMPolicy
from flashrl.envs import render_gif, render_ascii, Grid, Pong
set_seed(SEED:=1)

parser = argparse.ArgumentParser()
parser.add_argument('env_name')

ITERATIONS = 40
DURATION = 16
BS = 8*1024
N_HIDDEN = 128
LOG = False
PRINT_METRICS = ('loss',)# 'policy_loss', 'value_loss', 'entropy_loss')
SHOW_DATA = ('acts', 'rewards', 'dones', 'logprob', 'entropy', 'value')
DEVICE = 'cuda'  # even 'cpu' is surprisingly fast...
DTYPE = 'bfloat16'  # ...with float32
ENV_KWARGS = {'pong': {'size_x': 16, 'size_y': 8, 'max_dy': 1},
              'grid': {'size': 8}}
env = Pong(n_envs=2*BS, device=DEVICE, dtype=DTYPE).reset(seed=SEED)
model = LSTMPolicy(env, n_hidden=N_HIDDEN).to(device=env.device, dtype=env.dtype)
learn = Learner(env, model)
learn(ITERATIONS, DURATION, BS, HPARAMS.copy(), log=LOG, print_metrics=PRINT_METRICS, pbar_desc='done')
obs, data = learn.rollout(2*DURATION, SEED, extra_attrs=SHOW_DATA)
env.close()
render_gif('pong.gif', obs, obs_max=env.obs_max, fps=4, data=data)
render_ascii(obs, obs_max=env.obs_max, fps=8, data=data)
#print_table(data)  # Uncomment to print table
