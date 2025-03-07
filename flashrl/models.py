import math
import torch


class LSTMPolicy(torch.nn.Module):
    def __init__(self, env, n_hidden=128):
        super().__init__()
        self.encoder = torch.nn.Linear(math.prod(env.obs.shape[1:]), n_hidden)
        self.decoder = torch.nn.Linear(n_hidden, env.n_acts)
        self.value_head = torch.nn.Linear(n_hidden, 1)
        self.lstm = torch.nn.LSTMCell(n_hidden, n_hidden)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name: torch.nn.init.constant_(param, 0)
            elif 'weight' in name: torch.nn.init.orthogonal_(param, 1)

    def forward(self, x, state=None, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        x = self.encoder(x.view(len(x), -1)).relu()
        h, c = self.lstm(x, state)
        value = self.value_head(h)[:, 0]
        x = self.decoder(h)
        act = torch.multinomial(x.softmax(dim=-1), 1).byte().squeeze() if act is None else act
        x = x - x.logsumexp(dim=-1, keepdim=True)
        logprob = x.gather(-1, act[..., None].long())[..., 0]
        entropy = -(x * x.softmax(dim=-1)).sum(-1) if with_entropy else None
        return act, logprob, entropy, value, (h, c)
