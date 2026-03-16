from collections import defaultdict
from mini_poker.agents.batch_kernel_smoothed_methodical_agent import BatchKernelSmoothedMethodicalAgent


class GrokAgent(BatchKernelSmoothedMethodicalAgent):
    def __init__(self, game, **kwargs):
        super().__init__(game, **kwargs)
        self.gradient_buffer = defaultdict(lambda: defaultdict(float))
        self.epoch_counter = 0
        self.pairs_per_epoch = self.deck_size * (self.deck_size - 1)

    def train(self, print_period=1000):
        for iteration, (card1, card2) in enumerate(self.game.iter_uniformly(self.epochs)):
            if iteration % print_period == 0:
                self.print_progress(iteration)

            self._accumulate_grad(card1, card2)

            self.epoch_counter += 1
            if self.epoch_counter % self.pairs_per_epoch == 0:
                self._apply_batch_and_center()
                self.epoch_counter = 0  # reset for next epoch

    def _apply_batch_and_center(self):
        """Apply accumulated gradients + center EVERY infoset once per epoch."""
        for infoset, action_updates in self.gradient_buffer.items():
            for action, total_step in action_updates.items():
                self.update_logit(infoset, action, total_step)  # still clips
            self.softmax_update(infoset)
            self.center_logits(infoset)

        self.gradient_buffer.clear()

    def center_logits(self, infoset):
        """
        Subtracts the mean from all logits in an infoset so they sum to zero.
        This prevents logits from drifting together toward the bounds.
        """
        logits_dict = self.logits.get(infoset)
        if not logits_dict:
            return

        actions = list(logits_dict.keys())
        n = len(actions)
        if n <= 1:
            # If there's only one action, it must be 0 to sum to 0
            if n == 1:
                self.set_logit(infoset, actions[0], 0.0)
            return

        # Calculate current average
        avg_logit = sum(logits_dict.values()) / n

        # Subtract average from each logit
        for action in actions:
            centered_val = logits_dict[action] - avg_logit
            # set_logit handles the clipping to self.logit_bound
            self.set_logit(infoset, action, centered_val)
