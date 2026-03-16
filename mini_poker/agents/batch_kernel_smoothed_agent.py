import random
from collections import defaultdict
from mini_poker.agents.kernel_smoothed import KernelSmoothedAgent
from mini_poker.agents.base_agent import Infoset


class BatchKernelSmoothedAgent(KernelSmoothedAgent):
    """
    Subclass that accumulates all updates over a full epoch (all card combinations)
    and applies them in a single batch update at the end.
    """

    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        # Buffer to store accumulated updates: {infoset: {action: cumulative_update}}
        self.gradient_buffer = defaultdict(lambda: defaultdict(float))

    def train(self, print_period=1000):
        """
        Training loop:
        1. Accumulate updates for all card combinations.
        2. Apply updates and center logits once per epoch.
        """
        for iteration, (card1, card2) in enumerate(self.game.iter_uniformly(self.epochs)):
            if iteration % print_period == 0:
                self.print_progress(iteration)
            self._accumulate_grad(card1, card2)
            self._apply_batch_and_clear()

    def _update_gradient_buffer(self, actions, action_values, target_card, history, epsilon=1e-4):
        kernel = self._get_kernel_weights(target_card)

        # Spread updates via kernel, but store in buffer instead of updating logits
        for card_idx in range(self.deck_size):
            weight = kernel[card_idx]
            if weight < epsilon:
                continue

            infoset = Infoset(card_idx, history)
            probs = self.get_policy(infoset)
            baseline = sum(probs[a] * action_values[a] for a in actions)

            # Accumulate in buffer
            for action in actions:
                advantage = action_values[action] - baseline
                update_step = self.lr * advantage * weight
                self.gradient_buffer[infoset][action] += update_step

    def _accumulate_grad(self, card1, card2):
        explore = (random.random() < self.explore_proba)
        if explore:
            visited_infosets = self.sample_random_trajectory(card1, card2)
        else:
            visited_infosets = self.sample_trajectory(card1, card2)

        for (target_card, history), reach_prob in visited_infosets.items():
            actions = self.game.tree[history]
            action_values = self.get_action_values(actions, history, card1, card2)
            self._update_gradient_buffer(actions, action_values, target_card, history)

    def _apply_batch_and_clear(self):
        """
        Applies the accumulated gradients, centers logits, and clears buffer.
        If your 'epoch' in iter_uniformly represents one full pass of the deck,
        you apply it here. If iter_uniformly yields 1 pair per 'epoch',
        you might want to wrap this in a conditional based on deck_size.
        """
        for infoset, action_updates in self.gradient_buffer.items():
            for action, total_step in action_updates.items():
                # We reuse the logic from previous prompts:
                # update + clip + centering
                self.update_logit(infoset, action, total_step)

            # Ensure the policy is refreshed after the batch update
            self.softmax_update(infoset)

        # Clear buffer for the next epoch
        self.gradient_buffer.clear()
