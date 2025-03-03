from flashrl import HPARAMS, set_seed, print_table, Learner, LSTMPolicy
from flashrl.envs import render_gif, render_ascii, Grid
set_seed(SEED:=1)

ITERATIONS = 40
DURATION = 16
BS = 8*1024
N_HIDDEN = 128
#HPARAMS['lr'] = 1e-2
SIZE = 8
LOG = False
PRINT_METRICS = ('loss',)# 'policy_loss', 'value_loss', 'entropy_loss')
SHOW_DATA = ('acts', 'rewards', 'dones', 'logprob', 'entropy', 'value')
DEVICE = 'cuda'  # even 'cpu' is surprisingly fast...
DTYPE = 'bfloat16'  # ...with float32

env = Grid(n_envs=2*BS, size=SIZE, device=DEVICE, dtype=DTYPE).reset(seed=SEED)
model = LSTMPolicy(env, n_hidden=N_HIDDEN).to(device=env.device, dtype=env.dtype)
learn = Learner(env, model)
learn(ITERATIONS, DURATION, BS, HPARAMS.copy(), log=LOG, print_metrics=PRINT_METRICS)  # Training happens here!
obs, data = learn.rollout(DURATION, SEED, extra_attrs=SHOW_DATA)
env.close()
render_gif('gameplay.gif', obs, obs_max=env.obs_max, fps=2, data=data)
render_ascii(obs, obs_max=env.obs_max, fps=4, data=data)
#print_table(data)  # Uncomment to print table
