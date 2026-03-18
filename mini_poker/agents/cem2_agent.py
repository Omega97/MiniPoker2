import numpy as np
from mini_poker.agents.cem_agent import CEMAgent


class EnhancedCEMAgent(CEMAgent):
    """
    Subclass of CEMAgent that introduces Entropy Regularization
    and Gradient Clipping to the update rule.
    """

    def __init__(self, *args, entropy_beta=0.01, grad_clip=1.0, **kwargs):
        """
        :param entropy_beta: Strength of entropy regularization (higher = more exploration).
        :param grad_clip: Maximum absolute value for a single logit update.
        """
        self.entropy_beta = entropy_beta
        self.grad_clip = grad_clip
        super().__init__(*args, **kwargs)

    def _update_rule(self, actions, action_values, infoset):
        """
        Modified Update Rule:
        1. Uses a baseline to calculate advantage.
        2. Adds an entropy bonus to encourage diverse strategies.
        3. Applies gradient clipping for numerical stability.
        """
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * action_values[a] for a in actions)

        # Calculate local entropy for this infoset: -sum(p * log(p))
        # We use this to push the agent away from 100% / 0% certainty too early.
        local_entropy = -sum(p * np.log(p + 1e-9) for p in probs.values())

        for action in actions:
            # Standard Advantage
            advantage = action_values[action] - baseline

            # Entropy Bonus: Encourages actions that have low probability if they
            # aren't significantly worse than the baseline.
            # bonus = -log(p) -> higher update if probability is low.
            entropy_bonus = -np.log(probs[action] + 1e-9) * self.entropy_beta

            # Total Update
            update_step = self.lr * (advantage + entropy_bonus)

            # Gradient Clipping: Prevents single massive rewards/losses
            # from destroying a learned policy in one hand.
            update_step = np.clip(update_step, -self.grad_clip, self.grad_clip)

            self.update_logit(infoset, action, update_step)

        # Recalculate probabilities
        self.softmax_update(infoset)
