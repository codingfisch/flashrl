import sys, time, random, platform
import torch
import plotille
import numpy as np

from .envs import key_maps, emoji_maps


def play(env, model=None, playable=False, steps=None, fps=4, obs='obs', dump=False, with_data=True, idx=0, **kwargs):
    key_map = key_maps[env.__class__.__name__.lower()]
    emoji_map = emoji_maps[env.__class__.__name__.lower()]
    if playable: print(f'Press {"".join(key_map)} to act, m for model act and q to quit')
    data, state = {}, None
    for i in range((10000 if playable else 64) if steps is None else steps):
        data.update({'step': i})
        render(getattr(env, obs)[idx], cursor_up=i and not dump, emoji_map=emoji_map, data=data if with_data else None)
        act = np.zeros(len(env.obs), dtype=np.uint8)
        if model is not None:
            o = torch.from_numpy(env.obs).to(device=model.decoder.weight.device, dtype=model.decoder.weight.dtype)
            with torch.no_grad(): logp, val, state = model(o, state=state)
            act = torch.multinomial(logp.exp(), 1)[:, 0].byte()
            logp = logp.gather(1, act[:, None].long())[:, 0]
            entropy = -(logp * logp.exp())
            data.update({'model act': act[idx], 'logp': logp[idx], 'entropy': entropy[idx], 'value': val[idx]})
            act = act.cpu().numpy()
        key = get_pressed_key() if playable else f'm{time.sleep(1 / fps)}'[:1]
        if key == 'q': break
        act[idx] = act[idx] if key == 'm' else key_map[key] if key in key_map else 0
        env.step(act, **kwargs)
        data.update({'act': act[idx], 'reward': env.rewards[idx], 'done': env.dones[idx]})


def render(ob, cursor_up=True, emoji_map=None, data=None):
    if cursor_up: print(f'\033[A\033[{len(ob)}A')
    ob = 23 * (ob - ob.min()) / (ob.max() - ob.min()) + 232 if emoji_map is None else ob
    for i, row in enumerate(ob):
        for o in row.tolist():
            print(f'\033[48;5;{f"{232 + o}m" if emoji_map is None else f"232m{emoji_map[o]}"}\033[0m', end='')
        if data is not None:
            if i < len(data):
                print(f'{list(data.keys())[i]}: {list(data.values())[i]:.3g}', end='     ')
        print()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_curve(array, label=None, height=8, width=65):
    fig = plotille.Figure()
    fig._height, fig._width = height, width
    fig.y_label = fig.y_label if label is None else label
    fig.scatter(list(range(len(array))), array)
    print('\n'.join(fig.show().split('\n')[:-2]))


def get_pressed_key():
    if platform.system() == 'Windows':
        import msvcrt
        key = msvcrt.getch()
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key
