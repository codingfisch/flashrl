import flashrl as frl
frl.set_seed(SEED:=1)

env = frl.envs.MultiGrid(n_agents=2**14).reset(SEED)  # try one of: Pong, Grid, MultiGrid!
learn = frl.Learner(env, hidden_size=128, lstm=True)  # faster with lstm=False and smaller hidden_size
curves = learn.fit(1, steps=16, desc='done')  # ,lr=1e-2, gamma=.99) set hparams here
frl.print_curve(curves['loss'], label='loss')
frl.play(env, learn.model, with_human=False)  # if env is MultiGrid, try obs='total_obs', with_total_obs=True
env.close()
