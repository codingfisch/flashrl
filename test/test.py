import torch
from flashrl.utils import print_ascii_curve, render_ascii, render_gif, print_table
from flashrl.models import LSTMPolicy
from flashrl.main import get_advantages, Learner
from flashrl.envs.grid import Grid


def test_print_ascii_curve():
    array = [1, 2, 3, 4, 5]
    print_ascii_curve(array)

def test_LSTMPolicy_init():
    env = Grid(n_agents=1, size=5)
    model = LSTMPolicy(env)
    assert isinstance(model, LSTMPolicy), 'Model initialization failed'
    assert model.actor.in_features == 128, 'Actor layer initialization failed'
    assert model.lstm.hidden_size == 128, 'LSTM layer hidden size is incorrect'

def test_learner_init():
    env = Grid(n_agents=1, size=5)
    learner = Learner(env)
    assert isinstance(learner.model, LSTMPolicy), 'Model initialization failed'

def test_learner_fit():
    env = Grid(n_agents=2**13, size=5)
    learner = Learner(env)
    metrics = learner.fit(iters=1, steps=10)
    assert 'loss' in metrics, 'Fit did not return metrics'

def test_print_table():
    env = Grid(n_agents=2**13, size=5)
    learner = Learner(env)
    learner.fit(iters=1, steps=10)
    print_table(learner)

def test_render_ascii():
    env = Grid(n_agents=2**13, size=5)
    learner = Learner(env)
    learner.fit(iters=1, steps=10)
    render_ascii(learner)

def test_render_gif():
    env = Grid(n_agents=2**13, size=5)
    learner = Learner(env)
    learner.fit(iters=1, steps=10)
    render_gif('test.gif', learner)

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
    test_print_table()
    test_render_ascii()
    test_render_gif()
    test_get_advantages()
