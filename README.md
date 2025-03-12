# flashrl
`flashrl` does RL with **millions of steps/second 💨 while being tiny**: ~200 lines of code

🛠️ `pip install flashrl` or clone the repo & `pip install -r requirements.txt`
  - If cloned (or if envs changed), compile: `python setup.py build_ext --inplace`

💡 `flashrl` will always be **tiny**: **Read the code** (+paste into LLM) to understand it!
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
`.fit` does RL with ~**10 million steps**: `40` iterations × `16` steps × `2**14` agents!

<p align="center">
  <img src="https://github.com/user-attachments/assets/62da23a8-4d30-41f8-8843-1267e43a8744">
</p>

**Run it yourself via `python train.py` and play against the AI** 🪄

<details>
  <summary><b>Click here</b>, to read a tiny doc 📑</summary>

`Learner` takes the arguments
- `env`: RL environment
- `model`: A `Policy` model
- `device`: Per default picks `mps` or `cuda` if available else `cpu`
- `dtype`: Per default `torch.bfloat16` if device is `cuda` else `torch.float32`
- `compile_no_lstm`: Speedup via `torch.compile` if `model` has no `lstm`
- `**kwargs`: Passed to the `Policy`, e.g. `hidden_size` or `lstm`

`Learner.fit` takes the arguments
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
- `lr`, `anneal_lr` & args of `ppo` after `bs`: Hyperparameters

The most important functions in `flashrl/utils.py` are
- `print_curve`: Visualizes the loss across the `iters`
- `play`: Plays the environment in the terminal and takes
  - `model`: A `Policy` model
  - `playable`: If `True`, allows you to act (or decide to let the model act)
  - `steps`: Number of steps
  - `fps`: Frames per second
  - `obs`: Argument of the env that should be rendered as observations
  - `dump`: If `True`, no frame refresh -> Frames accumulate in the terminal
  - `idx`: Agent index between `0` and `n_agents` (default: `0`)
</details>

## Environments 🕹️
**Each env is one Cython(=`.pyx`) file** in `flashrl/envs`. **That's it!**

To **add custom envs**, use `grid.pyx`, `pong.pyx` or `multigrid.pyx` as a **template**:
- `grid.pyx` for **single-agent** envs (~110 LOC)
- `pong.pyx` for **1 vs 1 agent** envs (~150 LOC)
- `multigrid.pyx` for **multi-agent** envs (~190 LOC)

| `Grid`                | `Pong`                                                                                                    | `MultiGrid`                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| Agent must reach goal | Agent must score | Agent must reach goal first | 
|![grid](https://github.com/user-attachments/assets/f51c9fea-0ab9-45a1-a52e-446cee9fc593)| ![pong](https://github.com/user-attachments/assets/e77332d4-a3f4-432a-b338-98a078fb7dfb)| ![multigrid](https://github.com/user-attachments/assets/bc67c5e5-e820-4cfe-875c-1e545fbddff3)|

## Acknowledgements 🙌
I want to thank
- [Joseph Suarez](https://github.com/jsuarez5341) for open sourcing RL envs in C(ython)! Star [PufferLib](https://github.com/PufferAI/PufferLib) ⭐
- [Costa Huang](https://github.com/vwxyzjn) for open sourcing high-quality single-file RL code! Star [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐

and last but not least...

<p align="center">
  <img src="https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif">
</p>
