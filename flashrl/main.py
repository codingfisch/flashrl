import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils import print_plot
HPARAMS = {'lr': .01, 'anneal_lr': True, 'gamma': .99, 'gae_lambda': .95, 'clip_coef': .1,
           'value_coef': .5, 'value_clip_coef': .1, 'entropy_coef': .01, 'max_grad_norm': .5, 'norm_adv': True}


class Learner:
    def __init__(self, game, model, precision='medium'):
        self.env = game
        self.model = model
        torch.set_float32_matmul_precision(precision)

    def __call__(self, iterations, bs, duration, hparams, target_kl=None, log=False, print_metrics=None):
        metrics_curves = []
        logger = SummaryWriter() if log else None
        lr, anneal_lr = hparams.pop('lr'), hparams.pop('anneal_lr')
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        pbar = tqdm(range(iterations), total=iterations)
        obs, values, acts, logprobs, np_rewards, np_dones = self.get_buffers(duration)
        hc = torch.zeros((2, self.env.n_envs, self.model.lstm.hidden_size), dtype=self.env.dtype, device=self.env.device)
        for i in pbar:
            opt.param_groups[0]['lr'] = lr * (1 - i / iterations) if anneal_lr else lr
            obs, values, acts, logprobs, rewards, dones = evaluate(self.env, self.model, duration, obs, values, acts, logprobs, np_rewards, np_dones, state=(hc[0], hc[1]))
            metrics = train(self.model, opt, obs, values, acts, logprobs, rewards, dones, bs=bs, **hparams)
            pbar.set_description(f'reward: {rewards.mean():.3f}')
            dt = time.time() - (t if i > 0 else pbar.start_t)
            pbar.set_postfix({'': f'{1e-6 * self.env.n_envs * duration / dt:.1f}million steps/s'})
            t = time.time()
            if log:
                for k, v in metrics.items():
                    logger.add_scalar(k, v, global_step=i)
                for name, param in self.model.named_parameters():
                    logger.add_histogram(name, param, global_step=i)
            metrics_curves.append(metrics)
            if target_kl is not None:
                if metrics['kl'] > target_kl: break
        metric_curves = {k: [m[k].item() for m in metrics_curves] for k in metrics_curves[0]}
        if print_metrics is not None:
            for k in print_metrics:
                print_plot(metric_curves[k], k)
        return metric_curves

    def get_buffers(self, duration):
        shape = (self.env.n_envs, duration)
        obs = torch.empty((*shape, *self.env.obs_shape), dtype=self.env.dtype, device=self.env.device)
        values = torch.empty(shape, dtype=self.env.dtype, device=self.env.device)
        acts = torch.empty(shape, dtype=torch.int32, device=self.env.device)
        logprobs = torch.empty(shape, dtype=self.env.dtype, device=self.env.device)
        np_rewards = np.empty(shape, dtype=np.float32)
        np_dones = np.empty(shape, dtype=np.float32)
        return obs, values, acts, logprobs, np_rewards, np_dones

    def rollout(self, duration, seed=None, idx=0, obs_attr='obs', extra_attrs=None):
        self.env.reset(seed)
        attrs = [obs_attr] + list(extra_attrs or ())
        data = {k: [] for k in attrs}
        state = None
        for _ in range(duration):
            o = self.env.get_tensor(self.env.obs)
            act, logprob, entropy, value, state = self.model(o, state, with_entropy='entropy' in attrs)
            for attr in attrs:
                if attr in ('act', 'logprob', 'entropy', 'value'):
                    x = {'act': act, 'logprob': logprob, 'entropy': entropy, 'value': value}[attr]
                    x = x.detach().float().cpu().numpy()
                else:
                    x = getattr(self.env, attr).copy()
                data[attr].append(x[idx])
            self.env.step(act.cpu().numpy())
        data = {k: np.stack(xs) for k, xs in data.items()}
        obs = data.pop(obs_attr)
        return obs, data


def evaluate(env, model, duration, obs, values, acts, logprobs, np_rewards, np_dones, state=None):
    for i in range(duration):
        o =  env.get_tensor(env.obs)
        with torch.no_grad():
            act, logp, _, value, state = model(o, state=state)
        obs[:, i] = o
        values[:, i] = value
        acts[:, i] = act
        logprobs[:, i] = logp
        np_rewards[:, i] = env.rewards
        np_dones[:, i] = env.dones
        env.step(act.cpu().numpy())
    return obs, values, acts, logprobs, env.get_tensor(np_rewards), env.get_tensor(np_dones)


def train(model, opt, obs, values, acts, logprobs, rewards, dones, bs, gamma=.99, gae_lambda=.95, clip_coef=.1,
          value_coef=.5, value_clip_coef=.1, entropy_coef=.01, max_grad_norm=.5, norm_adv=True):
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
        with torch.no_grad():
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
