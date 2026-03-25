from typing import Dict
from mini_poker.game import Infoset
from mini_poker.agents.cem2_agent import CounterfactualEMAgent


DECISIVE_ACTIONS = ('C', 'F')


class DecisiveAgent(CounterfactualEMAgent):

    def __init__(self,
                 game,
                 logit_bound=10.,
                 epochs=100,
                 lr=0.1,
                 rollout_samples=1,
                 explore_proba=1.,
                 max_sigma=1.0,
                 n_games_compare=10_000,
                 epsilon=1e-2,
                 ):
        self.epsilon = epsilon
        super().__init__(game,
                         logit_bound=logit_bound,
                         epochs=epochs,
                         lr=lr,
                         rollout_samples=rollout_samples,
                         explore_proba=explore_proba,
                         max_sigma=max_sigma,
                         n_games_compare=n_games_compare,
                         )

    def get_policy(self, infoset: Infoset) -> Dict[str, float]:
        """
        If the agent only plays game-ending moves then use only the highest-proba move.

        When probability mass on continuing actions (R, D, A, etc.) drops below epsilon,
        collapse the policy to a deterministic choice among decisive actions (C, F).
        """
        p = self.policy[infoset]

        # Sum probability of decisive actions that are actually available
        p_decisive = sum(p.get(a, 0.0) for a in DECISIVE_ACTIONS)
        p_continue = 1 - p_decisive

        if p_continue > self.epsilon:
            # Still exploring continuing actions → return normal policy
            return self.policy[infoset]
        else:
            # Collapse to deterministic decisive action
            # Find the decisive action with highest probability
            available_decisive = [a for a in DECISIVE_ACTIONS if a in p]
            if available_decisive:
                best_action = max(available_decisive, key=lambda a: p.get(a, 0.0))
                # Return deterministic policy: best_action = 1.0, others = 0.0
                return {a: 1.0 if a == best_action else 0.0 for a in p.keys()}
            else:
                # Fallback: no decisive actions available, return original policy
                return self.policy[infoset]
