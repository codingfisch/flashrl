import flashrl as frl
frl.set_seed(SEED:=1)

env = frl.envs.Pong(n_agents=2**14)  # replace Pong with Grid to try it (for MultiGrid use commented code block below)
learn = frl.Learner(env=env.reset(SEED), hidden_size=128, lstm=True)  # even faster with smaller hidden_size/lstm=False
curves = learn.fit(40, steps=16, pbar_desc='done')  # ,lr=1e-2, gamma=.99) you can modify hparams here
frl.print_ascii_curve(curves['loss'], label='loss')
learn.rollout(steps=64)
frl.render_ascii(learn, fps=10)
frl.render_gif('pong.gif', learn, fps=4)
#frl.print_table(learn)
env.close()

# env = frl.envs.MultiGrid(n_agents=2**14, n_agents_per_env=2)
# learn = frl.Learner(env=env.reset(SEED), hidden_size=128, lstm=True)
# curves = learn.fit(40, steps=16)
# frl.print_ascii_curve(curves['loss'], label='loss')
# data = learn.rollout(steps=16, extra_args_list=['total_obs'], with_total_obs=True)
# frl.render_ascii(learn, fps=4, obs=data['total_obs'])
# frl.render_gif('multigrid.gif', learn, fps=4, obs=data['total_obs'])
# env.close()
