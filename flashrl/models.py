import math
import torch


class LSTMPolicy(torch.nn.Module):
    def __init__(self, env, n_hidden=128, n_layers=1):
        super().__init__()
        self.encoder = torch.nn.Linear(math.prod(env.obs_shape), n_hidden)
        self.decoder = torch.nn.Linear(n_hidden, env.n_acts)
        self.value_head = torch.nn.Linear(n_hidden, 1)
        self.lstm = torch.nn.LSTM(n_hidden, n_hidden, num_layers=n_layers)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name: torch.nn.init.constant_(param, 0)
            elif 'weight' in name: torch.nn.init.orthogonal_(param, 1)

    def forward(self, x, state=None, act=None, with_entropy=None):
        with_entropy = act is not None if with_entropy is None else with_entropy
        x = self.encoder(x.view(len(x), -1)).relu()
        x = x.view(x.shape[0], 1, self.lstm.hidden_size)
        x, state = self.lstm(x.transpose(0, 1), state)
        x = x.transpose(0, 1).reshape(-1, self.lstm.hidden_size)
        value = self.value_head(x)[:, 0]
        x = self.decoder(x)
        act = torch.multinomial(x.softmax(dim=-1), 1).int().squeeze() if act is None else act
        x = x - x.logsumexp(dim=-1, keepdim=True)
        logprob = x.gather(-1, act[..., None].long())[..., 0]
        entropy = -(x * x.softmax(dim=-1)).sum(-1) if with_entropy else None
        return act, logprob, entropy, value, state
