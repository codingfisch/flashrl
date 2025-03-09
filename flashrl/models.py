import math
import torch


class Policy(torch.nn.Module):
    def __init__(self, env, hidden_size=128, lstm=False):
        super().__init__()
        self.encoder = torch.nn.Linear(math.prod(env.obs.shape[1:]), hidden_size)
        self.actor = torch.nn.Linear(hidden_size, env.n_acts)
        self.value_head = torch.nn.Linear(hidden_size, 1)
        self.lstm = cleanrl_init(torch.nn.LSTMCell(hidden_size, hidden_size)) if lstm else None

    def forward(self, x, state, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        h = self.encoder(x.view(len(x), -1)).relu()
        h, c = (h, None) if self.lstm is None else self.lstm(h, state)
        value = self.value_head(h)[:, 0]
        x = self.actor(h)
        act = torch.multinomial(x.softmax(dim=-1), 1)[:, 0] if act is None else act
        x = x - x.logsumexp(dim=-1, keepdim=True)
        logprob = x.gather(-1, act[..., None].long())[..., 0]
        entropy = -(x * x.softmax(dim=-1)).sum(-1) if with_entropy else None
        return act.byte(), logprob, entropy, value, (h, c)


def cleanrl_init(module):
    for name, param in module.named_parameters():
        if 'bias' in name: torch.nn.init.constant_(param, 0)
        elif 'weight' in name: torch.nn.init.orthogonal_(param, 1)
    return module
