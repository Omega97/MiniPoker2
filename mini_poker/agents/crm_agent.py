import numpy as np
from collections import defaultdict
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import Infoset
from itertools import permutations


class CRMAgent(BaseAgent):
    """
    Counterfactual Regret Minimization Agent.
    Following the architecture of CEM2Agent but utilizing
    Cumulative Regret and Regret Matching for policy updates.
    """

    def __init__(self,
                 game,
                 logit_bound=10.,
                 epochs=50,
                 lr=1.0,  # In CFR, 'lr' acts as a regret scaling factor
                 max_sigma=5.0,
                 momentum=0.9):  # Momentum is highly effective for CFR convergence

        self.epochs = epochs
        self.lr = lr
        self.max_sigma = max_sigma
        self.momentum = momentum

        # Buffers for CFR
        # cumulative_regret: sum of (action_value - node_value) over time
        self.cumulative_regret = defaultdict(lambda: defaultdict(float))
        # strategy_sum: used to compute the average strategy (the Nash Equilibrium)
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        super().__init__(game, logit_bound)

        # Precompute kernels for card abstraction/smoothing
        self._kernel_matrix = self._precompute_kernels()

    def _init_name(self):
        self.name = f"CRM({self.game.game_power},{self.game.deck_size})_e{self.epochs}_s{self.max_sigma:.1f}"

    def _precompute_kernels(self):
        """Standard Gaussian kernel lookup table for card smoothing."""
        matrix = []
        for card in range(self.deck_size):
            normalized_strength = card / (self.deck_size - 1)
            variance = self.max_sigma * (1.0 - normalized_strength)

            if variance < 1e-6:
                weights = np.zeros(self.deck_size)
                weights[card] = 1.0
            else:
                cards = np.arange(self.deck_size)
                weights = np.exp(-((cards - card) ** 2) / (2 * variance))
                weights /= np.sum(weights)
            matrix.append(weights)
        return np.array(matrix)

    def _regret_matching(self, infoset: Infoset):
        """
        Updates the policy based on cumulative regrets.
        Actions with higher positive regret get higher probability.
        """
        regrets = self.cumulative_regret[infoset]
        actions = self.game.tree[infoset.branch]

        # Get positive regrets
        pos_regrets = {a: max(0.0, regrets[a]) for a in actions}
        sum_pos_regret = sum(pos_regrets.values())

        if sum_pos_regret > 0:
            new_policy = {a: pos_regrets[a] / sum_pos_regret for a in actions}
        else:
            # Default to uniform if no positive regret exists
            new_policy = {a: 1.0 / len(actions) for a in actions}

        self.set_policy(infoset, new_policy)

    def _accumulate_regret(self, card1, card2):
        """
        Performs a traversal to calculate counterfactual values
        and updates cumulative regret buffers.
        """
        history = ""
        self._traverse(history, card1, card2, 1.0, 1.0)

    def _traverse(self, history, card1, card2, p1_reach, p2_reach):
        """Recursive tree traversal for CFR."""
        if history in self.game.terminals:
            # Return utility for both players
            return self.game.get_reward(history, card1, card2)

        player = len(history) % 2
        my_card = card1 if player == 0 else card2
        actions = self.game.tree[history]
        infoset = Infoset(my_card, history)

        # 1. Get current strategy
        strategy = self.get_policy(infoset)

        # 2. Compute action values (recursively)
        action_utilities = {}
        node_utility = 0.0

        for action in actions:
            if player == 0:
                child_util = self._traverse(history + action, card1, card2, p1_reach * strategy[action], p2_reach)
                action_utilities[action] = child_util[0]
            else:
                child_util = self._traverse(history + action, card1, card2, p1_reach, p2_reach * strategy[action])
                action_utilities[action] = child_util[1]

            node_utility += strategy[action] * action_utilities[action]

        # 3. Update Cumulative Regret (weighted by opponent reach probability)
        opp_reach = p2_reach if player == 0 else p1_reach
        kernel = self._kernel_matrix[my_card]

        for card_idx, weight in enumerate(kernel):
            if weight < 1e-4: continue

            sim_infoset = Infoset(card_idx, history)
            for action in actions:
                regret = (action_utilities[action] - node_utility) * opp_reach * weight
                self.cumulative_regret[sim_infoset][action] = (
                        self.momentum * self.cumulative_regret[sim_infoset][action] + self.lr * regret
                )

            # Update strategy sum (for average strategy calculation)
            my_reach = p1_reach if player == 0 else p2_reach
            for action in actions:
                self.strategy_sum[sim_infoset][action] += my_reach * strategy[action] * weight

        return (node_utility, -node_utility) if player == 0 else (-node_utility, node_utility)

    def train(self):
        print(f"\nTraining {self} using Regret Minimization...")
        for epoch in range(self.epochs):
            # Visit all card combinations
            for c1, c2 in permutations(range(self.deck_size), 2):
                self._accumulate_regret(c1, c2)

            # Update policies for all infosets based on new regrets
            for history in self.game.tree:
                for card in range(self.deck_size):
                    self._regret_matching(Infoset(card, history))

            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{self.epochs} | Entropy: {self.entropy():.4f}")

    def get_average_strategy(self):
        """
        CFR theory states the average strategy converges to Nash.
        This method computes that average from the strategy_sum buffer.
        """
        avg_policy = {}
        for infoset, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_policy[infoset] = {a: s / total for a, s in action_sums.items()}
            else:
                actions = self.game.tree[infoset.branch]
                avg_policy[infoset] = {a: 1.0 / len(actions) for a in actions}
        return avg_policy
