emoji_maps = {'grid': {0: '  ', 1: '🦠', 2: '🍪'},
              'pong': {0: '  ', 1: '🔲', 2: '🔴'},
              'multigrid': {0: '  ', 1: '🧱', 2: '🦠', 3: '🍪'}}
key_maps = {'grid': {'a': 1, 'd': 2, 'w': 3, 's': 4},
            'pong': {'w': 1, 's': 2},
            'multigrid': {'a': 1, 'd': 2, 'w': 3, 's': 4}}
from .grid import Grid
from .pong import Pong
from .multigrid import MultiGrid
