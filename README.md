# flashrl
RL in **seconds** 💨 with **~200 lines of code** (+ ~150 per env) 🤓

🛠️ No pip package yet! Install via
```bash
git clone https://github.com/codingfisch/flashrl
pip install torch Cython tensorboard plotille pillow tqdm
```
## Quick Start 🚀
1. Compile envs via `python setup.py build_ext --inplace`
2. Train via `python train.py`
3. See the magic unfold in the terminal 🪄

<p align="center">
  <img src="https://github.com/user-attachments/assets/6cc1277a-e6e6-4162-98fd-5b76505e9644">
</p>

## Usage 💡
`flashrl` will always be **short** → **Read the code** (+paste into ChatGPT) to fully understand it!

Here is a **minimal example** to get you started:

`flashrl` uses a `Learner` that holds an `env` and a `model` (default: `LSTMPolicy`)
```python
import flashrl as frl

learn = frl.Learner(env=frl.envs.Pong(n_agents=2**14))
curves = learn.fit(40, steps=16, pbar_desc='done')
frl.print_ascii_curve(curves['loss'], label='loss')
frl.render_ascii(learn, fps=10)
learn.env.close()
```
`.fit` triggers RL with
- **40** iterations...
- ...**16** steps per iteration...
- ...in `Pong` holding `2**14`=**16384** agents

resulting in training with (40 * 16 * 16384=)~**10 million steps**!

<details>
  <summary><b>Click here</b>, to read a tiny doc 📑</summary>

`.fit` takes the arguments
- `iters`: Number of iterations
- `steps`: Number of steps in `rollout`
- `pbar_desc`: Progress bar description (default: `'reward'`)
- `log`: If `True`, `tensorboard` logging is enabled 
  - run `tensorboard --logdir=runs`and visit `http://localhost:6006` in the browser!
- `lr`, `anneal_lr`, `target_fl` + all args of `ppo`: Hyperparameters

Take a look at `train.py` to see how to use the `utils`-functions
- `print_ascii_curve`: Visualizes the loss across the `iters`
- `render_ascii`: Shows data of the last `rollout` in the terminal
- `render_gif`: Shows the same, saved as a GIF
- `print_table`: Shows a table of values, acts, logprobs, reward and dones of the last `rollout`
</details>

## Environments 🕹️
**Each env is one Cython(=`.pyx`) file** in `flashrl/envs`. **That's it!**

To **add custom envs**, use `grid.pyx`, `pong.pyx` or `multigrid.pyx` as a **template**:
- `grid.pyx` for **single-agent** envs
- `pong.pyx` for **1 vs. 1 agent** envs (AlphaZero-style)
- `multigrid.pyx` for **multi-agent** envs

| `Grid`                | `Pong`                  | `MultiGrid`                 |
|-----------------------|-------------------------|-----------------------------|
| Agent must reach goal | Good old pong (1 vs. 1) | Agent must reach goal first |
| ![grid](https://github.com/user-attachments/assets/e3f84b2f-e8f8-4fc5-a483-b5711489a7af) | ![pong](https://github.com/user-attachments/assets/ed462fe4-0edc-404c-af83-d634f23015fd) | ![multigrid](https://github.com/user-attachments/assets/7fd502f0-447f-4dd1-a8a1-e22044502c90) |

## Acknowledgements 🙌
I want to thank
- [Joseph Suarez](https://github.com/jsuarez5341) for open source RL envs in C(ython)! Star [PufferLib](https://github.com/PufferAI/PufferLib) ⭐
- [Costa Huang](https://github.com/vwxyzjn) for open source high-quality single-file RL code! Star [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐

and last but not least...

<p align="center">
  <img src="https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif">
</p>
