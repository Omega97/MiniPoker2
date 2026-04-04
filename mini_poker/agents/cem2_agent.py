import random
import numpy as np
from typing import Dict
from collections import defaultdict
from mini_poker.game import MiniPoker, Infoset, State
from mini_poker.agents.base_agent import BaseAgent


class CounterfactualEMAgent(BaseAgent):
    """
    Counterfactual Expectation Maximization Agent.
    A streamlined version of the CounterfactualAgent.
    Removes momentum and batching for a direct 'REINFORCE-style'
    update to the tabular policy.
    """
    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=100,
                 lr=0.1,
                 rollout_samples=1,
                 explore_proba=0.1,
                 kernel_size=2.,
                 ):
        self.epochs = epochs
        self.lr = lr
        self.rollout_samples = rollout_samples
        self.kernel_size = kernel_size
        self.gradient_buffer = defaultdict(lambda: defaultdict(float))
        self.terminal_paths = None
        self.visited_infosets = None
        self._kernel_matrix = None
        self._path_pointer = 0

        super().__init__(game, logit_bound, explore_proba=explore_proba, epochs=epochs)
        self._precompute_kernels()
        self._init_terminal_paths()

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        lr_digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{lr_digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_r{self.rollout_samples}"
        self.name += f"_p{self.explore_proba * 100:.0f}"
        self.name += f"_s{self.kernel_size * 100:.0f}"

    def _init_terminal_paths(self):
        self.terminal_paths = list(self.game.terminals.keys())
        random.shuffle(self.terminal_paths)

    def sample_policy_trajectory(self, state: State) -> dict:
        """Perform random trajectory and returns dict of info."""
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = state.card_p1 if player == 0 else state.card_p2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            probs = self.get_policy(infoset)
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]
            history += action
        return visited

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"
        return bar

    def get_baseline(self, action_values: dict, infoset: Infoset) -> float:
        # Calculate the baseline (expected value) for the current policy
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * v for a, v in action_values.items())
        return baseline

    def _precompute_kernels(self, epsilon=1e-6):
        """Builds a lookup table for kernels for every possible card."""
        matrix = []
        for card in range(self.deck_size):
            # Use your existing logic once for each card
            variance = self._get_variance(card)

            if variance <= epsilon:
                weights = np.zeros(self.deck_size)
                weights[card] = 1.0
            else:
                cards = np.arange(self.deck_size)
                weights = np.exp(-((cards - card) ** 2) / (2 * variance))
                weights /= np.sum(weights)

            matrix.append(weights)
        self._kernel_matrix = np.array(matrix)

    def _get_variance(self, card: int) -> float:
        """
        Calculates variance which decreases as card value increases.
        Variance is kernel_size at card 0 and 0 at the highest card.
        """
        # Linear decay: sigma^2 goes from kernel_size to 0
        normalized_strength = card / (self.deck_size - 1)
        return self.kernel_size * (1.0 - normalized_strength)

    def get_kernel_weights(self, center_card: int):
        """Replaces heavy calculation with a simple lookup."""
        return self._kernel_matrix[center_card]

    def print_kernel_weights(self):
        """Prints a visualization of the kernel weights for each card."""
        print(f"\nKernel Weight Distribution (kernel_size={self.kernel_size})")
        print(f"{'Card':<5} | {'Weights (Card 0' + ' ' * (self.deck_size * 5) + 'Card N)':<10}")
        print("-" * (15 + self.deck_size * 6))
        for card in range(self.deck_size):
            weights = self.get_kernel_weights(card)
            # Format weights into a string of small bars or percentages
            weight_str = " ".join([f"{w:4.2f}" if w > 0.01 else " .  " for w in weights])
            variance = self._get_variance(card)
            print(f"{card:<5} | {weight_str}  (σ²={variance:.2f})")

    def get_action_values(self, infoset, state):
        """Perform rollouts for each action, then return average values."""
        player = infoset.get_current_player()
        rewards = self.get_average_rewards(infoset).copy()
        counts = self.get_visit_counts(infoset)
        actions = self.game.tree[infoset.branch]

        for action in actions:
            n_new_samples = max(0, self.rollout_samples - counts[action])
            if n_new_samples:
                temp_state = state
                temp_state = temp_state.perform_action(action)
                new_rewards = [self.rollout(temp_state)[player] for _ in range(self.rollout_samples)]
                new_avg = sum(new_rewards) / len(new_rewards)
                rewards[action] = (rewards[action] * counts[action] + new_avg * n_new_samples) / self.rollout_samples

        return rewards

    def get_next_path(self):
        """Cycle through the shuffled terminal paths."""
        path = self.terminal_paths[self._path_pointer]
        self._path_pointer = (self._path_pointer + 1) % len(self.terminal_paths)
        return path

    def sample_random_trajectory(self, state: State) -> Dict[Infoset, float]:
        trajectory = self.get_next_path()
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = state.card_p1 if player == 0 else state.card_p2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            actions = self.game.tree[history]
            action = trajectory[len(history)]
            reach_prob *= (1.0 / len(actions))
            history += action
        return visited

    def get_posterior(self, infoset: Infoset):
        """
        Re-uses your Bayesian inference logic to determine
        what the opponent is likely holding.
        """
        game = self.game
        num_cards = game.deck_size
        opponent_probs = np.zeros(num_cards)

        for card_opp in range(num_cards):
            if card_opp == infoset.card:
                continue

            reach_prob = 1.0
            temp_hist = ""
            my_player_index = len(infoset.branch) % 2

            for i, action in enumerate(infoset.branch):
                acting_player = i % 2
                # If it was the opponent's turn in the past
                if acting_player != my_player_index:
                    prev_infoset = Infoset(card_opp, temp_hist)
                    # We use the current policy to see how likely they were to do this
                    probs = self.get_policy(prev_infoset)
                    reach_prob *= probs.get(action, 0.0)
                temp_hist += action
            opponent_probs[card_opp] = reach_prob

        total_weight = np.sum(opponent_probs)
        if total_weight > 0:
            return opponent_probs / total_weight
        else:
            # Fallback to uniform if history is 'impossible'
            post = np.ones(num_cards) / (num_cards - 1)
            post[infoset.card] = 0
            return post

    # --- Core Training Algorithm ---

    def training_epoch(self):
        """
        Training loop:
        1. Sample trajectories.
        2. Accumulate updates for all card combinations.
        3. Apply updates and center logits once per epoch.
        """
        self._uniform_sampling_step()
        self._accumulate_grad()
        self._apply_batch_and_clear()
        self.normalize_logits()

    def _uniform_sampling_step(self):
        """Visit all card combinations once."""
        for card1, card2 in self.game.iter_uniformly_over_hands():
            state = State(card1, card2, "")
            with self.train_context():
                self.sample_trajectory_from_root(state)

    def _accumulate_grad(self):
        """Apply feedback to the gradient buffer."""
        assert len(self.trajectories_cache)

        for trajectory in self.trajectories_cache:
            for state in trajectory.get_state_history():
                infoset = state.get_current_player_infoset()
                action_values = self.get_action_values(infoset, state)
                action_counts = self.get_visit_counts(infoset)
                self._update_gradient_buffer(infoset, action_values, action_counts)

        self.clear_trajectories_cache()

    def _update_gradient_buffer(self,
                                infoset: Infoset,
                                action_values: dict,
                                action_counts: dict,
                                epsilon=1e-4):
        """Update rule on logit gradient buffer."""
        assert infoset in self.policy, f"{infoset}"

        kernel = self.get_kernel_weights(infoset.card)
        values = list(action_values.values())
        std = float(np.std(values))
        if std <= epsilon:
            return

        # Spread updates via kernel, but store in buffer instead of updating logits
        for card_idx in range(self.deck_size):
            weight = kernel[card_idx]
            if weight < epsilon:
                continue

            # baseline = self.get_baseline(action_values, infoset)
            n_moves = len(action_values)
            best_action = max(action_values, key=action_values.get)
            other_update = -1 / (n_moves - 1)

            # Accumulate in buffer
            for action, value in action_values.items():
                n_visits = max(action_counts.get(action), self.rollout_samples)
                update_step = (1 if action == best_action else other_update) * self.lr * weight / n_visits
                self.gradient_buffer[infoset][action] += update_step

    def _apply_batch_and_clear(self):
        """
        Applies the accumulated gradients, centers logits, and clears buffer.
        If your 'epoch' in iter_uniformly represents one full pass of the deck,
        you apply it here. If iter_uniformly yields 1 pair per 'epoch',
        you might want to wrap this in a conditional based on deck_size.
        """
        for infoset, action_updates in self.gradient_buffer.items():
            for action, total_step in action_updates.items():
                # We reuse the logic from previous prompts: update + clip + centering
                self.update_logit(infoset, action, total_step)

            # Ensure the policy is refreshed after the batch update
            self.softmax_update(infoset)

        # Clear buffer for the next epoch
        self.gradient_buffer.clear()
