from mini_poker.agents.base_agent import BaseAgent, Infoset


class RegretMatchingAgent(BaseAgent):
    def __init__(self, game):
        self.regrets = {}
        self.policy_sum = {}  # Tracks the cumulative average strategy
        super().__init__(game)

    def _init_policy(self):
        """Initialize regrets, current policy, and the average policy sum."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.regrets[infoset] = {a: 0.0 for a in actions}
                self.policy_sum[infoset] = {a: 0.0 for a in actions}
                # Initial current policy is uniform
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def get_average_policy(self, infoset: Infoset) -> dict:
        """Returns the converged average strategy for an infoset."""
        sum_weights = sum(self.policy_sum[infoset].values())
        if sum_weights > 0:
            return {a: val / sum_weights for a, val in self.policy_sum[infoset].items()}
        else:
            # Fallback to uniform if never visited
            actions = self.game_tree[infoset.branch]
            return {a: 1.0 / len(actions) for a in actions}

    def train(self, epochs=1000, rollout_samples=5):
        """Training loop with Average Strategy tracking."""
        for epoch, (card1, card2) in enumerate(self.game.iter_uniformly(epochs)):
            # 1. Sample trajectory and get reach probabilities
            # visited_infosets should return {infoset: reach_prob}
            visited_infosets = self.sample_trajectories(card1, card2)

            for (card, history), reach_prob in visited_infosets.items():
                infoset = Infoset(card, history)
                player = len(history) % 2
                actions = self.game_tree[history]

                # 2. Update the Average Policy (The "Stable" Strategy)
                # Weight by reach_prob and epoch (linear CFR) for faster convergence
                current_p = self.get_policy(infoset)
                for a in actions:
                    self.policy_sum[infoset][a] += current_p[a] * reach_prob * (epoch + 1)

                # 3. Calculate Counterfactual Values
                action_values = {}
                for action in actions:
                    rewards = self.evaluate_action(history, action, card1, card2, rollout_samples)
                    action_values[action] = rewards[player]

                # 4. Update Regrets
                ev = sum(current_p[a] * action_values[a] for a in actions)
                for a in actions:
                    regret = action_values[a] - ev
                    self.regrets[infoset][a] += regret

                # 5. Update Current Policy for the next iteration
                self._regret_matching_update(infoset)

    def _regret_matching_update(self, infoset):
        """Standard Regret Matching for the 'current' strategy."""
        regrets = self.regrets[infoset]
        positive_regrets = {a: max(0.0, r) for a, r in regrets.items()}
        total_pos_regret = sum(positive_regrets.values())

        if total_pos_regret > 0:
            new_policy = {a: r / total_pos_regret for a, r in positive_regrets.items()}
        else:
            actions = self.game_tree[infoset.branch]
            new_policy = {a: 1.0 / len(actions) for a in actions}

        self.set_policy(infoset, new_policy)
