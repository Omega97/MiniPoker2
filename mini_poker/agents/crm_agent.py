import random
from time import time
from mini_poker.agents.base_agent import BaseAgent, Infoset


class CounterfactualRegretMinimizationAgent(BaseAgent):
    """
    Counterfactual Regret Minimization (MCCFR) agent.

    This is a Monte-Carlo Counterfactual Regret Minimization implementation:
    - Samples full trajectories (policy-based or uniform random for exploration)
    - Accumulates linear-weighted average strategy (converges to Nash equilibrium)
    - Computes counterfactual action values via rollouts from the current infoset
    - Updates cumulative regrets with advantages
    - Applies standard regret-matching to obtain the instantaneous strategy
    - Uses the instantaneous strategy for sampling and action selection (standard CFR practice)

    The average strategy (get_average_policy) is the theoretically guaranteed equilibrium strategy,
    but get_action continues to use the instantaneous (regret-matched) policy for consistency
    with other agents in the framework.
    """

    def __init__(self,
                 game,
                 epochs=10_000,
                 rollout_samples=5,
                 explore_proba=0.0):
        self.epochs = epochs
        self.rollout_samples = rollout_samples
        self.explore_proba = explore_proba

        # CFR-specific storage
        self.regrets = {}
        self.policy_sum = {}

        # Progress tracking (same as CounterfactualAgent)
        self.t0 = None
        self.epoch_start = None

        super().__init__(game)
        self._init_regret_structures()
        self._init_name()

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        self.name += f"_r{self.rollout_samples}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def _init_regret_structures(self):
        """Initialize regrets and average-strategy accumulators for every infoset."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.regrets[infoset] = {a: 0.0 for a in actions}
                self.policy_sum[infoset] = {a: 0.0 for a in actions}
                # Current policy is already uniform (handled by BaseAgent)

    def get_average_policy(self, infoset: Infoset) -> dict:
        """Return the average strategy (the strategy that converges to Nash)."""
        sum_weights = sum(self.policy_sum[infoset].values())
        if sum_weights > 0:
            return {a: val / sum_weights for a, val in self.policy_sum[infoset].items()}
        actions = self.game.tree[infoset.branch]
        return {a: 1.0 / len(actions) for a in actions}

    def _regret_matching_update(self, infoset):
        """Update current policy from positive cumulative regrets (regret matching)."""
        regrets = self.regrets[infoset]
        positive_regrets = {a: max(0.0, r) for a, r in regrets.items()}
        total_pos_regret = sum(positive_regrets.values())

        if total_pos_regret > 0:
            new_policy = {a: r / total_pos_regret for a, r in positive_regrets.items()}
        else:
            actions = self.game.tree[infoset.branch]
            new_policy = {a: 1.0 / len(actions) for a in actions}

        self.set_policy(infoset, new_policy)

    def sample_trajectory(self, card1, card2) -> dict:
        """Sample a trajectory using the current (regret-matched) policy."""
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            probs = self.policy[infoset]
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]
            history += action
        return visited

    def sample_random_trajectory(self, card1, card2) -> dict:
        """Uniform random trajectory (used during exploration)."""
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            actions = self.game.tree[history]
            action = random.choice(actions)
            reach_prob *= (1.0 / len(actions))
            history += action
        return visited

    def print_progress(self, epoch, t_long=600):
        """Identical progress reporting as CounterfactualAgent for UI consistency."""
        time_left = None
        tot_epochs = self.epochs * self.deck_size * (self.deck_size - 1)

        if self.t0 is None:
            self.t0 = time()
            self.epoch_start = 0
        else:
            time_elapsed = time() - self.t0

            if time_elapsed > t_long:
                self.t0 = time()
                self.epoch_start = epoch
                time_elapsed = 0.001

            recent_progress = epoch - self.epoch_start
            if time_elapsed > 0:
                speed = recent_progress / time_elapsed
                remaining_epochs = tot_epochs - epoch
                if speed > 0:
                    time_left = remaining_epochs / speed

        p = epoch / tot_epochs
        out = f"\r{epoch:6})  {p:.2%}  S = {self.entropy():.4f}   F = {self.best_card_fold_index():.4f}"
        if time_left is not None:
            out += f"   time left: {time_left / 60:.0f} min"
        print(out, end='')

    def train(self, print_period=1000):
        """
        MCCFR training loop (one full pass over the deck per epoch).

        For each sampled infoset on the trajectory:
          1. Update the linear-weighted average strategy
          2. Estimate counterfactual values for every legal action (via rollouts)
          3. Compute advantages and accumulate regrets
          4. Apply regret matching to obtain the new instantaneous strategy
        """
        for iteration, (card1, card2) in enumerate(self.game.iter_uniformly(self.epochs)):

            if iteration % print_period == 0:
                self.print_progress(iteration)

            # 1. Sample trajectory (policy or exploratory)
            explore = (random.random() < self.explore_proba)
            if explore:
                visited_infosets = self.sample_random_trajectory(card1, card2)
            else:
                visited_infosets = self.sample_trajectory(card1, card2)

            # 2. Process every infoset visited on this trajectory
            for (card, history), reach_prob in visited_infosets.items():
                infoset = Infoset(card, history)
                player = len(history) % 2
                actions = self.game.tree[history]

                # Update average strategy (linear CFR weighting)
                current_p = self.get_policy(infoset)
                for a in actions:
                    self.policy_sum[infoset][a] += current_p[a] * reach_prob * (iteration + 1)

                # Counterfactual value estimation for every action
                action_values = {}
                for action in actions:
                    rewards = self.evaluate_action(history, action, card1, card2, self.rollout_samples)
                    action_values[action] = rewards[player]

                # Baseline (current expected value)
                ev = sum(current_p[a] * action_values[a] for a in actions)

                # Accumulate regrets (advantage = counterfactual value difference)
                for a in actions:
                    regret = action_values[a] - ev
                    self.regrets[infoset][a] += regret

                # Regret matching → new instantaneous strategy
                self._regret_matching_update(infoset)
