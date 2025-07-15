import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .models import Policy
DEVICE = 'mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


class Learner:
    def __init__(self, env, model=None, device=None, dtype=None, jit=False, **kwargs):
        self.env = env
        self.device = DEVICE if device is None else device
        self.dtype = dtype if dtype is not None else torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.model = Policy(self.env, **kwargs).to(self.device, self.dtype) if model is None else model
        if jit: self.model = torch.compile(self.model, fullgraph=True, mode='reduce-overhead')
        self._data, self._np_data, self._rollout_state, self._ppo_state = None, None, None, None

    def fit(self, iters=40, steps=16, lr=.01, bs=None, anneal_lr=True, opt=None, log=False, desc=None, stop_func=None,
            **hparams):
        if opt is None:
            opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5, betas=(.9, .999), weight_decay=0)
        bs = bs or len(self.env.obs) // 2
        self.setup_data(steps)
        logger = SummaryWriter() if log else None
        pbar = tqdm(range(iters), total=iters)
        curves = []
        for i in pbar:
            opt.param_groups[0]['lr'] = lr * (1 - i / iters) if anneal_lr else lr
            self.rollout(steps)
            losses = ppo(self.model, opt, bs=bs, **self._data, **hparams)
            if desc: pbar.set_description(
                f'{desc}: {losses[desc] if desc in losses else self._data[desc].mean():.3f}')
            if i: pbar.set_postfix_str(
                f'{1e-6 * self._data["act"].numel() * pbar.format_dict["rate"]:.1f}M steps/s')
            if log:
                for k, v in losses.items(): logger.add_scalar(k, v, global_step=i)
                for name, param in self.model.named_parameters(): logger.add_histogram(name, param, global_step=i)
            curves.append(losses)
            if stop_func is not None:
                if stop_func(**self._data, **losses): break
        return {k: [m[k].item() for m in curves] for k in curves[0]}

    def setup_data(self, steps):
        x = torch.zeros((len(self.env.obs), steps), dtype=self.dtype, device=self.device)
        obs = torch.zeros((*x.shape, *self.env.obs.shape[1:]), dtype=self.dtype, device=self.device)
        self._data = {'obs': obs, 'logp': x.clone(), 'value': x, 'act': x.clone().byte()}
        self._np_data = {'reward': x.char().cpu().numpy(), 'done': x.char().cpu().numpy()}
        if self.model.lstm is not None:
            zeros = torch.zeros((len(obs), self.model.decoder.in_features), dtype=self.dtype, device=self.device)
            self._rollout_state = (zeros, zeros.clone())
            self._data.update({'state': zeros[:, None].repeat(1, steps, 2)})

    def rollout(self, steps, state=None, extra_args_list=None, **kwargs):
        state = self._rollout_state if state is None else state
        if steps != (0 if self._data is None else self._data['obs'].shape[1]): self.setup_data(steps)
        extra_data = {} if extra_args_list is None else {k: [] for k in extra_args_list}
        for i in range(steps):
            if self.model.lstm is not None:
                self._data['state'][:, i, :state[0].shape[1]] = state[0]
                self._data['state'][:, i, state[0].shape[1]:] = state[1]
            o = self.to_torch(self.env.obs)
            with torch.no_grad():
                logp, value, state = self.model(o, state=state)
            act = torch.multinomial(logp.exp(), 1)
            self._data['obs'][:, i] = o
            self._data['logp'][:, i] = logp.gather(1, act)[:, 0]
            act = act[:, 0].byte()
            self._data['value'][:, i] = value
            self._data['act'][:, i] = act
            self._np_data['reward'][:, i] = self.env.rewards
            self._np_data['done'][:, i] = self.env.dones
            for k in extra_data: extra_data[k].append(self.to_torch(getattr(self.env, k).copy()))
            self.env.step(act.cpu().numpy(), **kwargs)
        self._data.update({k: self.to_torch(v) for k, v in self._np_data.items()})
        return {k: torch.stack(v, dim=1) for k, v in extra_data.items()}

    def to_torch(self, x, non_blocking=True):
        return torch.from_numpy(x).to(device=self.device, dtype=self.dtype, non_blocking=non_blocking)


def ppo(model, opt, obs, value, act, logp, reward, done, bs=2**13, gamma=.99, gae_lambda=.95, clip_coef=.1,
        value_coef=.5, value_clip_coef=.1, entropy_coef=.01, max_grad_norm=.5, norm_adv=True, state=None):
    advs = get_advantages(value, reward, done, gamma=gamma, gae_lambda=gae_lambda)
    obs, value, act, logp, advs = [xs.view(-1, bs, *xs.shape[2:]) for xs in [obs, value, act, logp, advs]]
    state = state if state is None else state.view(-1, bs, *state.shape[2:])
    returns = advs + value
    metrics, metric_keys = [], ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl']
    for i, (o, old_value, a, old_logp, adv, ret) in enumerate(zip(obs, value, act, logp, advs, returns)):
        hc = None if state is None else (state[i, :, :state.shape[-1] // 2], state[i, :, state.shape[-1] // 2:])
        new_logp, val, _ = model(o, state=hc)
        new_logp = new_logp.gather(-1, a[:, None].long())[:, 0]
        logratio = new_logp - old_logp
        ratio = logratio.exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if norm_adv else adv
        policy_loss = torch.max(-adv * ratio, -adv * ratio.clip(1 - clip_coef, 1 + clip_coef)).mean()
        if value_clip_coef:
            v_clipped = old_value + (val - old_value).clip(-value_clip_coef, value_clip_coef)
            value_loss = .5 * torch.max((val - ret) ** 2, (v_clipped - ret) ** 2).mean()
        else:
            value_loss = .5 * ((val - ret) ** 2).mean()
        entropy = -(new_logp * new_logp.exp()).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        kl = ((ratio - 1) - logratio).mean()
        metrics.append([loss, policy_loss, value_loss, entropy, kl])
    return {k: torch.stack([values[i] for values in metrics]).mean() for i, k in enumerate(metric_keys)}


def get_advantages(value, reward, done, gamma=.99, gae_lambda=.95):  # see arxiv.org/abs/1506.02438 eq. (16)-(18)
    advs = torch.zeros_like(value)
    not_done = 1. - done
    for t in range(1, done.shape[1]):
        delta = reward[:, -t] + gamma * value[:, -t] * not_done[:, -t] - value[:, -t - 1]
        advs[:, -t-1] = delta + gamma * gae_lambda * not_done[:, -t] * advs[:, -t]
    return advs
