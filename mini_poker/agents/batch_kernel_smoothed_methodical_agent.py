import random
from typing import Dict
from mini_poker.agents.base_agent import Infoset
from mini_poker.agents.batch_kernel_smoothed_agent import BatchKernelSmoothedAgent


class BatchKernelSmoothedMethodicalAgent(BatchKernelSmoothedAgent):
    """
    An agent that combines:
    1. Methodical traversal of every terminal node in the tree.
    2. Kernel smoothing across similar cards.
    3. Batch updates applied only after the full game space is explored.
    """
    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        # Precompute and shuffle all paths once at start
        self.terminal_paths = list(self.game.terminals.keys())
        random.shuffle(self.terminal_paths)
        self._path_pointer = 0

    def get_next_path(self):
        """Cycle through the shuffled terminal paths."""
        path = self.terminal_paths[self._path_pointer]
        self._path_pointer = (self._path_pointer + 1) % len(self.terminal_paths)
        return path

    def sample_random_trajectory(self, card1, card2) -> Dict[Infoset, float]:
        trajectory = self.get_next_path()
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            actions = self.game.tree[history]
            action = trajectory[len(history)]
            reach_prob *= (1.0 / len(actions))
            history += action
        return visited
