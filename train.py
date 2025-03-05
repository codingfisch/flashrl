import flashrl as frl
frl.set_seed(SEED:=1)

DEVICE, DTYPE = 'cuda', 'bfloat16'  # for CPU use 'cpu', 'float32'
learn = frl.Learner(env=frl.envs.Pong(n_envs=16*1024, device=DEVICE, dtype=DTYPE).reset(SEED))
# learn.model = frl.LSTMPolicy(learn.env, n_hidden=128).to(DEVICE, DTYPE)  # uncomment to modify model (e.g. n_hidden)
curves = learn.fit(40, steps=16, lr=1e-2, log=False, pbar_desc='done')  # modify hparams here (e.g. gamma)
learn.env.close()
frl.print_curves(curves, keys='loss')
frl.render_ascii(learn._data['obs'][0], obs_max=learn.env.obs_max, fps=4)#, data=data)
frl.render_gif('gameplay.gif', learn._data['obs'][0], obs_max=learn.env.obs_max, fps=2)#, data=data)
