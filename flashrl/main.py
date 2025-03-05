import torch
from tqdm import tqdm
from time import time
from torch.utils.tensorboard import SummaryWriter

from .models import LSTMPolicy


class Learner:
    def __init__(self, env, model=None, precision='medium'):
        self.env = env
        self.model = LSTMPolicy(self.env).to(self.env.device, self.env.dtype) if model is None else model
        self._data, self._np_data = None, None
        torch.set_float32_matmul_precision(precision)

    def fit(self, iters=40, steps=16, lr=.01, anneal_lr=True, log=False, pbar_desc=None, target_kl=None, **hparams):
        self.setup_data(steps)
        metrics_curves = []
        logger = SummaryWriter() if log else None
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        pbar = tqdm(range(iters), total=iters)
        hc = torch.zeros((2, self.env.n_envs, self.model.lstm.hidden_size), dtype=self.env.dtype, device=self.env.device)
        for i in pbar:
            t0 = time()
            opt.param_groups[0]['lr'] = lr * (1 - i / iters) if anneal_lr else lr
            self.rollout(steps, state=(hc[0], hc[1]))
            metrics = ppo(self.model, opt, **self._data, **hparams)
            pbar.set_description(f'{pbar_desc}: {self._data[pbar_desc+"s"].mean():.3f}')
            pbar.set_postfix_str(f'{1e-6 * self.env.n_envs * steps / (time() - t0):.1f}million steps/s')
            if log:
                for k, v in metrics.items(): logger.add_scalar(k, v, global_step=i)
                for name, param in self.model.named_parameters(): logger.add_histogram(name, param, global_step=i)
            metrics_curves.append(metrics)
            if target_kl is not None:
                if metrics['kl'] > target_kl: break
        return {k: [m[k].item() for m in metrics_curves] for k in metrics_curves[0]}

    def setup_data(self, duration):
        values = torch.empty((self.env.n_envs, duration), dtype=self.env.dtype, device=self.env.device)
        obs = torch.empty((*values.shape, *self.env.obs_shape), dtype=self.env.dtype, device=self.env.device)
        self._data = {'obs': obs, 'values': values, 'acts': values.clone().byte(), 'logprobs': values.clone()}
        self._np_data = {'rewards': values.float().cpu().numpy(), 'dones': values.float().cpu().numpy()}

    def rollout(self, duration, state=None):
        for i in range(duration):
            o = self.to_torch(self.env.obs)
            with torch.no_grad():
                act, logp, _, value, state = self.model(o, state=state)
            self._data['obs'][:, i] = o
            self._data['values'][:, i] = value
            self._data['acts'][:, i] = act
            self._data['logprobs'][:, i] = logp
            self._np_data['rewards'][:, i] = self.env.rewards
            self._np_data['dones'][:, i] = self.env.dones
            self.env.step(act.cpu().numpy())
        self._data.update({k: self.to_torch(v, non_blocking=True) for k, v in self._np_data.items()})

    def to_torch(self, x, device=None, dtype=None, **kwargs):
        return torch.from_numpy(x).to(device=device or self.env.device, dtype=dtype or self.env.dtype, **kwargs)


def ppo(model, opt, obs, values, acts, logprobs, rewards, dones, bs=8192, gamma=.99, gae_lambda=.95,
        clip_coef=.1, value_coef=.5, value_clip_coef=.1, entropy_coef=.01, max_grad_norm=.5, norm_adv=True):
    advs = get_advantages(values, rewards, dones, gamma=gamma, gae_lambda=gae_lambda)
    obs, values, acts, logprobs, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, values, acts, logprobs, advs]]
    returns = advs + values
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl', 'clip_frac']
    state = None
    for o, old_value, act, old_logp, adv, ret in zip(obs, values, acts, logprobs, advs, returns):
        _, logp, entropy, value, state = model(o, state=state, act=act)
        state = (state[0].detach(), state[1].detach())
        logratio = logp - old_logp
        ratio = logratio.exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
        policy_loss = torch.max(-adv * ratio, -adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).mean()
        if value_clip_coef:
            v_clipped = old_value + (value - old_value).clip(-value_clip_coef, value_clip_coef)
            value_loss = .5 * torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
            value_loss = .5 * ((value - ret) ** 2).mean()
        entropy = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        kl, clip_frac = ((ratio - 1) - logratio).mean(), ((ratio - 1).abs() > clip_coef).float().mean()
        metrics.append([loss, policy_loss, value_loss, entropy, kl, clip_frac])
    return {k: torch.stack([values[i] for values in metrics]).mean() for i, k in enumerate(metric_keys)}


def get_advantages(values, rewards, dones, gamma=.99, gae_lambda=.95):  # see arxiv.org/abs/1506.02438 eq. (16)-(18)
    advs = torch.zeros_like(values)
    not_dones = 1. - dones
    for t in range(1, dones.shape[1]):
        delta = rewards[:, -t] + gamma * values[:, -t] * not_dones[:, -t] - values[:, -t-1]
        advs[:, -t-1] = delta + gamma * gae_lambda * not_dones[:, -t] * advs[:, -t]
    return advs
