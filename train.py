import flashrl as frl
frl.set_seed(SEED:=1)

env = frl.envs.Pong(n_envs=2**14, device='cuda', dtype='bfloat16')  # for CPU use 'cpu', 'float32'
learn = frl.Learner(env=env.reset(SEED))
#learn.model = frl.LSTMPolicy(learn.env, n_hidden=128).to(env.device, env.dtype)  # uncomment to modify model
curves = learn.fit(40, steps=16, pbar_desc='done')  # modify hparams here (e.g. lr, gamma...)
frl.print_ascii_curves(curves, keys='loss')
frl.render_ascii(learn, fps=8)
frl.render_gif('gameplay.gif', learn, fps=4)
#frl.print_table(learn)
env.close()
