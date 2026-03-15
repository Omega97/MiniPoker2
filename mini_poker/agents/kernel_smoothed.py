import random
import numpy as np
from mini_poker.agents.counterfactual_agent import CounterfactualAgent, Infoset


class KernelSmoothedAgent(CounterfactualAgent):
    """
    Counterfactual agent that uses a Gaussian kernel to smooth updates
    across similar hand values within the same game branch.
    """
    def __init__(self, game, max_sigma=2.0, **kwargs):
        self.max_sigma = max_sigma
        super().__init__(game, **kwargs)
        self._kernel_matrix = self._precompute_kernels()

    def _init_name(self):
        super()._init_name()
        self.name += f"_s{self.max_sigma * 100:.0f}"

    def _precompute_kernels(self):
        """Builds a lookup table for kernels for every possible card."""
        matrix = []
        for card in range(self.deck_size):
            # Use your existing logic once for each card
            variance = self._get_variance(card)

            if variance <= 1e-6:
                weights = np.zeros(self.deck_size)
                weights[card] = 1.0
            else:
                cards = np.arange(self.deck_size)
                weights = np.exp(-((cards - card) ** 2) / (2 * variance))
                weights /= np.sum(weights)

            matrix.append(weights)
        return np.array(matrix)

    def _get_variance(self, card: int) -> float:
        """
        Calculates variance which decreases as card value increases.
        Variance is max_sigma at card 0 and 0 at the highest card.
        """
        # Linear decay: sigma^2 goes from max_sigma to 0
        normalized_strength = card / (self.deck_size - 1)
        return self.max_sigma * (1.0 - normalized_strength)

    def _get_kernel_weights(self, center_card: int):
        """Replaces heavy calculation with a simple lookup."""
        return self._kernel_matrix[center_card]

    def train(self, print_period=1000):
        """
        Training loop that spreads the advantage update across
        similar cards using the kernel.
        """
        for epoch, (card1, card2) in enumerate(self.game.iter_uniformly(self.epochs)):

            if epoch % print_period == 0:
                self.print_progress(epoch)

            explore = (random.random() < self.explore_proba)
            visited_infosets = self.sample_trajectories(card1, card2, explore=explore)

            for (target_card, history), reach_prob in visited_infosets.items():
                player = len(history) % 2
                actions = self.game.tree[history]

                # 1. Evaluate actions for the specific cards drawn
                action_values = {}
                for action in actions:
                    rewards = self.evaluate_action(history, action, card1, card2, self.rollout_samples)
                    action_values[action] = rewards[player]

                # 2. Update logits for ALL cards in this history branch using the kernel
                kernel = self._get_kernel_weights(target_card)

                for card_idx in range(self.deck_size):
                    weight = kernel[card_idx]
                    if weight < 1e-4:
                        continue  # Optimization: skip negligible updates

                    infoset = Infoset(card_idx, history)
                    probs = self.get_policy(infoset)

                    # Calculate baseline for the specific card being updated
                    baseline = sum(probs[a] * action_values[a] for a in actions)

                    for action in actions:
                        advantage = action_values[action] - baseline
                        # Apply update weighted by kernel similarity
                        update_step = self.lr * advantage * weight
                        self.update_logit(infoset, action, update_step)

                    self.softmax_update(infoset)

    def print_kernel_weights(self):
        """Prints a visualization of the kernel weights for each card."""
        print(f"\nKernel Weight Distribution (max_sigma={self.max_sigma})")
        print(f"{'Card':<5} | {'Weights (Card 0' + ' ' * (self.deck_size * 5) + 'Card N)':<10}")
        print("-" * (15 + self.deck_size * 6))

        for card in range(self.deck_size):
            weights = self._get_kernel_weights(card)
            # Format weights into a string of small bars or percentages
            weight_str = " ".join([f"{w:4.2f}" if w > 0.01 else " .  " for w in weights])
            variance = self._get_variance(card)
            print(f"{card:<5} | {weight_str}  (σ²={variance:.2f})")
