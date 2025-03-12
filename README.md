# flashrl
RL library that trains with **millions of steps/second 💨 while being tiny**: ~200 lines of code(+150 per env)

🛠️ `pip install flashrl` or clone the repo and `pip install -r requirements.txt`

🛠️ If cloned (or when envs were changed/added), compile: `python setup.py build_ext --inplace`

💡 `flashrl` will always be **short**: **Read the code** (+paste into LLM) to understand it!
## Quick Start 🚀
`flashrl` uses a `Learner` that holds an `env` and a `model` (default: `Policy` with LSTM)

```python
import flashrl as frl

learn = frl.Learner(frl.envs.Pong(n_agents=2**14))
curves = learn.fit(40, steps=16, desc='done')
frl.print_curve(curves['loss'], label='loss')
frl.play(learn.env, learn.model, fps=8)
learn.env.close()
```
`.fit` triggers RL with ~**10 million steps**: `40` iterations with `16` steps with `2**14` agents!

<p align="center">
  <img src="https://github.com/user-attachments/assets/6cc1277a-e6e6-4162-98fd-5b76505e9644">
</p>

**Run it yourself via `python train.py` and play against the AI** 🪄

<details>
  <summary><b>Click here</b>, to read a tiny doc 📑</summary>

`Learner` takes
- `env`: RL environment
- `model`: A `Policy` model
- `device`: Per default picks `mps` if available, elif `cuda` else `cpu`
- `dtype`: Per default `torch.bfloat16` if device is `cuda` else `torch.float32`
- `compile_no_lstm`: Speedup via `torch.compile` if `model` has no `lstm`
- `**kwargs`: Passed to the `Policy`, e.g. `hidden_size` or `lstm`

`.fit` takes the arguments
- `iters`: Number of iterations
- `steps`: Number of steps in `rollout`
- `desc`: Progress bar description (e.g. `'reward'`)
- `log`: If `True`, `tensorboard` logging is enabled 
  - run `tensorboard --logdir=runs`and visit `http://localhost:6006` in the browser!
- `stop_func`: Function that stops training if it returns `True` e.g.

```python
...
def stop(kl, **kwargs):
  return kl > .1

curves = learn.fit(40, steps=16, stop_func=stop)
...
```
- `lr`, `anneal_lr` & all args of `ppo`: Hyperparameters

Use `train.py` and take a look into `flashrl/utils.py` to understand how
- `print_curve`: Visualizes the loss across the `iters`
- `play`: Plays the environment in the terminal and takes
  - `model`: A `Policy` model
  - `playable`: If `True`, allows you to act (or decide to let the model act)
  - `steps`: Number of steps
  - `fps`: Frames per second
  - `obs`: Argument of the env that should be rendered as observation
  - `dump`: If `True`, no frame refresh -> Frames accumulate in the terminal
  - `idx`: Picks an agent between `0` and `n_agents` (default: `0`)
</details>

## Environments 🕹️
**Each env is one Cython(=`.pyx`) file** in `flashrl/envs`. **That's it!**

To **add custom envs**, use `grid.pyx`, `pong.pyx` or `multigrid.pyx` as a **template**:
- `grid.pyx` for **single-agent** envs
- `pong.pyx` for **1 vs 1 agent** envs (AlphaZero-style)
- `multigrid.pyx` for **multi-agent** envs

| `Grid`                | `Pong`                                                                                                    | `MultiGrid`                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Agent must reach goal ![grid](https://github.com/user-attachments/assets/e3f84b2f-e8f8-4fc5-a483-b5711489a7af)| Agent must score ![pong](https://github.com/user-attachments/assets/ed462fe4-0edc-404c-af83-d634f23015fd) | Agent must reach goal first ![multigrid](https://github.com/user-attachments/assets/7fd502f0-447f-4dd1-a8a1-e22044502c90)                                                                  |

## Acknowledgements 🙌
I want to thank
- [Joseph Suarez](https://github.com/jsuarez5341) for open sourcing RL envs in C(ython)! Star [PufferLib](https://github.com/PufferAI/PufferLib) ⭐
- [Costa Huang](https://github.com/vwxyzjn) for open sourcing high-quality single-file RL code! Star [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐

and last but not least...

<p align="center">
  <img src="https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif">
</p>
