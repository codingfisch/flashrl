import pandas as pd
from flashrl import HPARAMS, set_seed, Learner, LSTMPolicy
from flashrl.envs import render_gif, render_ascii, Grid
set_seed(SEED:=1)

ITERATIONS = 50
BS = 8*1024
DURATION = 32
N_HIDDEN = 128
#HPARAMS['lr'] = 1e-2
LOG = False
PRINT_METRICS = ('loss', 'policy_loss', 'value_loss', 'entropy_loss')
SHOW_DATA = ('acts', 'rewards', 'dones', 'logprob', 'entropy', 'value')
DEVICE = 'cuda'  # even 'cpu' is surprisingly fast...
DTYPE = 'bfloat16'  # ...with float32

env = Grid(n_envs=BS // 2, size=8, device=DEVICE, dtype=DTYPE).reset(seed=SEED)
model = LSTMPolicy(env, n_hidden=N_HIDDEN).to(device=env.device, dtype=env.dtype)
learn = Learner(env, model)
learn(ITERATIONS, BS, DURATION, HPARAMS.copy(), log=LOG, print_metrics=PRINT_METRICS)  # Training happens here!
obs, data = learn.rollout(DURATION, SEED, extra_attrs=SHOW_DATA)
render_gif('gameplay.gif', obs, obs_max=env.obs_max, fps=2, data=data)
render_ascii(obs, obs_max=env.obs_max, fps=4, data=data)
env.close()
# print(pd.DataFrame(data))
