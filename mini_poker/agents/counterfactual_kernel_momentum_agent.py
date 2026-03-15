from collections import defaultdict
from mini_poker.agents.kernel_smoothed import KernelSmoothedAgent


class MomentumSmoothedAgent(KernelSmoothedAgent):
    """
    Extends KernelSmoothedAgent with momentum and zero-centered logit updates.
    """
    def __init__(self, game, momentum=0.9, **kwargs):
        self.momentum = momentum
        # velocity[infoset][action] stores the moving average of updates
        self.velocity = defaultdict(lambda: defaultdict(float))
        super().__init__(game, **kwargs)

    def _init_name(self):
        super()._init_name()
        self.name += f"_m{self.momentum * 100:.0f}"

    def update_logit(self, infoset, action, update_step):
        """
        Updates logit using momentum and then centers the entire
        infoset to maintain sum(logits) = 0.
        """
        # 1. Update velocity (momentum)
        v = self.velocity[infoset][action]
        new_v = self.momentum * v + (1 - self.momentum) * update_step
        self.velocity[infoset][action] = new_v

        # 2. Apply the momentum-weighted step to the logit
        current_val = self.get_logit(infoset, action)
        self.set_logit(infoset, action, current_val + new_v)

        # 3. Mean-center all logits at this infoset to prevent boundary drift
        self.center_logits(infoset)

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
