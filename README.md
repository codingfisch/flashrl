# flashrl
RL in seconds 💨 with ~200 lines of code (+ ~100 per env) 🤓

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
  <img src="https://github.com/user-attachments/assets/4ac3f6f0-972b-4ee8-bf93-ba4c905b3b92">
</p>

## Usage 💡
`flashrl` uses a `Learner` that holds an `env` and a `model`
```python
from flashrl import HPARAMS, Learner, LSTMPolicy
from flashrl.envs import Grid

BS = 8*1024  # (mini-)batch size
env = Grid(n_envs=2*BS, size=8, device='cuda').reset()
learn = Learner(env=env,
                model=LSTMPolicy(env).to(device=env.device, dtype=env.dtype))
learn(40, duration=16, bs=BS, hparams=HPARAMS.copy(), print_metrics=('loss',))
```
The last line triggers RL
- with 40 iterations...
- ...with 16 steps run per iteration...
- ...with `Grid` containing 16384(=`2*BS`) envs (in C)

resulting in ~10 million(=40 * 16 * 16384) steps run!

<details>
  <summary><b>Click here</b>, to see some more usage explanations 📑</summary>
The call to Learner takes the arguments

- `iterations`: Number of iterations
- `duration`: Number of steps in `evaluate`
- `bs`: Batch size (also called minibatch size in RL)
- `hparams`: Dictionary of hyperparameters
- `log`: If `True`, `tensorboard` logging is enabled 
  - run `tensorboard --logdir=runs`...
  - ...and visit `http://localhost:6006` in the browser!
- `print_metrics`: Tuple containing metrics that will be printed(=ASCII plot)
  - Possible metrics: `'loss'`, `'policy_loss'`, `'value_loss'`, `'entropy_loss'`, `'kl'`, `'clip_frac'`
- `target_kl`: Target KL (Kullback-Leibler) divergence

Take a look at `train.py` to understand how to use
- `render_ascii` to show a rollout in the terminal
- `render_gif` to save a GIF of the rollout
- `print_table` to print a table of the rollout
</details>

Wanna bring your **own env**? **Rewrite it in C** with `flashrl/envs/grid` as a **template**!

`flashrl` will always be short. Just read the code (+paste into ChatGPT) to understand!

## Acknowledgements 🙌
I want to thank
- [Joseph Suarez](https://github.com/jsuarez5341) for open source RL envs in C(ython)! Star [PufferLib](https://github.com/PufferAI/PufferLib) ⭐
- [Costa Huang](https://github.com/vwxyzjn) for open source high-quality single-file RL code! Star [cleanrl](https://github.com/vwxyzjn/cleanrl) ⭐

and last but not least...

<p align="center">
  <img src="https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif">
</p>
