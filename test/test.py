import torch
from flashrl.utils import play, print_curve
from flashrl.models import Policy
from flashrl.main import get_advantages, Learner
from flashrl.envs.grid import Grid


def test_print_ascii_curve():
    array = [1, 2, 3, 4, 5]
    print_curve(array)

def test_LSTMPolicy_init():
    env = Grid(n_agents=1, size=5)
    model = Policy(env, lstm=True)
    assert isinstance(model, Policy), 'Model initialization failed'
    assert model.actor.in_features == 128, 'Actor layer initialization failed'
    assert model.lstm.hidden_size == 128, 'LSTM layer hidden size is incorrect'

def test_learner_init():
    env = Grid(n_agents=1, size=5)
    learn = Learner(env)
    assert isinstance(learn.model, Policy), 'Model initialization failed'

def test_learner_fit():
    env = Grid(n_agents=2**13, size=5)
    learn = Learner(env)
    losses = learn.fit(iters=1)
    assert 'loss' in losses, 'Fit did not return metrics'

def test_play():
    env = Grid(n_agents=2**13, size=5)
    learn = Learner(env)
    learn.fit(iters=1)
    play(env, learn.model, steps=16)

def test_get_advantages():
    values = torch.rand(1, 10)
    rewards = torch.rand(1, 10)
    dones = torch.zeros(1, 10)
    advantages = get_advantages(values, rewards, dones)
    assert advantages is not None, 'Advantages calculation failed'
    assert advantages.shape == values.shape, 'Advantages shape mismatch'

if __name__ == '__main__':
    test_print_ascii_curve()
    test_LSTMPolicy_init()
    test_learner_init()
    test_learner_fit()
    test_play()
    test_get_advantages()
