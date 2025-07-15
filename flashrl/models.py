import math
import torch


class Policy(torch.nn.Module):
    def __init__(self, env, hidden_size=128, lstm=False):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Flatten(),
                                           torch.nn.Linear(math.prod(env.obs.shape[1:]), hidden_size),
                                           torch.nn.ReLU())
        self.decoder = torch.nn.Linear(hidden_size, env.n_acts + 1)
        self.lstm = cleanrl_init(torch.nn.LSTMCell(hidden_size, hidden_size)) if lstm else None

    def forward(self, x, state):
        h = self.encoder(x)
        h, c = (h, None) if self.lstm is None else self.lstm(h, state)
        logits_value = self.decoder(h)
        logits = logits_value[:, :-1]
        return logits - logits.logsumexp(dim=-1, keepdim=True), logits_value[:, -1], (h, c)


def cleanrl_init(module):
    for name, param in module.named_parameters():
        if 'bias' in name: torch.nn.init.constant_(param, 0)
        elif 'weight' in name: torch.nn.init.orthogonal_(param, 1)
    return module
