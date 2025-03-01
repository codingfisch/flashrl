# flashrl
RL in seconds 💨 with ~200 lines of code (+ 100 for the RL env) 🤓

🛠️ Not a pip package yet! For now install via
```bash
git clone https://github.com/codingfisch/flashrl
pip install torch Cython tensorboard plotille pillow pandas tqdm
```
## Usage 💡
1. Compile RL envs via `python setup.py build_ext --inplace`
2. Train via `python train.py`
3. See the magic unfold in your terminal 🪄

![train](https://github.com/user-attachments/assets/46db059f-8958-4f13-8ff9-b4a98580e054)

## Acknowledgements 🙌
I want to thank
- [Joseph Suarez](https://github.com/jsuarez5341) for open sourcing the idea to write RL envs in C(ython)! Give [PufferLib](https://github.com/PufferAI/PufferLib) a star ⭐
- [Costa Huang](https://github.com/vwxyzjn) for open sourcing high-quality single-file RL code! Give [cleanrl](https://github.com/vwxyzjn/cleanrl) a star ⭐

and last but not least...

![snoop](https://media1.tenor.com/m/ibYVxrR2hOgAAAAC/well-done.gif)
