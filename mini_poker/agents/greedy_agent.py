import math
from mini_poker.agents.base_agent import BaseAgent


class GreedyAgent(BaseAgent):

    def __init__(self,
                 trained_agent,
                 n_visits=100,
                 k_greed=2.0,
                 logit_bound=10.0):
        """
        An agent that transforms a trained policy into a greedy-biased policy.

        :param trained_agent: An instance of CRMAgent or BaseAgent with training data.
        :param n_visits: Threshold for self.reward_counts to apply the boost.
        :param k_greed: The constant added to the logit of the best-EV action.
        """
        # Inherit the game and basic parameters from the trained agent
        super().__init__(trained_agent.game, logit_bound=logit_bound)

        self.n_visits = n_visits
        self.k_greed = k_greed

        # 1. Inherit existing knowledge
        self.policy = trained_agent.policy.copy()
        self.average_reward = trained_agent.average_reward
        self.reward_counts = trained_agent.reward_counts

        # 2. Perform the global policy transformation
        self._compute_greedy_policy()

    def _compute_greedy_policy(self):
        """
        Iterates through all infosets to apply the EV-boost logic all at once.
        """
        p_min = math.exp(-self.logit_bound)
        for infoset in list(self.logits.keys()):
            # A) Init logits from policy
            self.logits[infoset] = {a: math.log(max(p, p_min)) for a, p in self.get_policy(infoset).items()}
            self.center_logits(infoset)

            # B) Check if this infoset meets the visit threshold
            # Summing visits across all actions in this infoset
            total_visits = sum(self.reward_counts[infoset].values())

            if total_visits >= self.n_visits:
                # Find the action with the highest average reward
                avg_rewards = self.average_reward[infoset]
                if avg_rewards:
                    best_action = max(avg_rewards, key=avg_rewards.get)

                    # C) Add k_greed to the highest-EV move
                    current_logit = self.get_logit(infoset, best_action)
                    self.set_logit(infoset, best_action, current_logit + self.k_greed)

            # D) Re-compute probabilities using softmax
            self.softmax_update(infoset)
