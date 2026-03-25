import random
from mini_poker.game import Infoset, State
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
import numpy as np


class GreedyAgent(CounterfactualEMAgent):
    def __init__(self,
                 game,
                 logit_bound=10.,
                 epochs=100,
                 lr=0.1,
                 rollout_samples=1,
                 explore_proba=1.,
                 max_sigma=1.0,
                 n_games_compare=10_000,
                 n_samples=10,
                 p_exploit=0.1,
                 ):
        """
        GreedyAgent always plays the action with highest expected value.
        """
        self.n_samples = n_samples
        super().__init__(game,
                         logit_bound=logit_bound,
                         epochs=epochs,
                         lr=lr,
                         rollout_samples=rollout_samples,
                         explore_proba=explore_proba,
                         max_sigma=max_sigma,
                         n_games_compare=n_games_compare,
                         )

    def rollout(self, state: State):
        """Returns (p1_reward, p2_reward)"""
        while state.branch not in self.game.terminals:
            player = len(state.branch) % 2
            card = state.card_p1 if player == 0 else state.card_p2
            infoset = Infoset(card, state.branch)
            action = self.base_get_action(infoset)
            state.branch += action
        return self.game.get_reward(state)

    def base_get_action(self, infoset: Infoset) -> str:
        """Necessary to avoid recursion."""
        return super().get_action(infoset)

    def get_action(self, infoset: Infoset) -> str:
        """
        Return the action with highest expected value.
        Samples opponent cards uniformly (avoids recursion through get_policy).
        """
        if random.random() < self.explore_proba:
            # Get available actions from raw policy dict (NO get_policy() call!)
            probs = self.policy[infoset]
            actions = list(probs.keys())
            player = infoset.get_current_player()
            values = [self.bayesian_evaluate_action(infoset, a, n_samples=self.n_samples)[player] for a in actions]
            return actions[np.argmax(values)]
        else:
            return super().get_action(infoset)
