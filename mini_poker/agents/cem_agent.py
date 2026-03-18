import random
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Dict
from collections import defaultdict
from mini_poker.game import MiniPoker
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import Infoset
from mini_poker.training.evaluation import evaluate_agents


class CEMAgent(BaseAgent):
    """
    Counterfactual Expectation Maximization Agent.
    A streamlined version of the CounterfactualAgent.
    Removes momentum and batching for a direct 'REINFORCE-style'
    update to the tabular policy.
    """
    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=50,
                 lr=0.01,
                 rollout_samples=1,
                 explore_proba=1.,
                 max_sigma=2.0,
                 n_games_compare=20_000,
                 ):
        self.epochs = epochs
        self.lr = lr
        self.rollout_samples = rollout_samples
        self.explore_proba = explore_proba
        self.max_sigma = max_sigma
        self.n_games_compare = n_games_compare
        self.t0 = None
        self.epoch_start = None
        self.compare_agent = None
        self.ev_list = []

        super().__init__(game, logit_bound)
        self.gradient_buffer = defaultdict(lambda: defaultdict(float))
        self.terminal_paths = None
        self._kernel_matrix = self._precompute_kernels()
        self._path_pointer = 0
        self._init_terminal_paths()

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_r{self.rollout_samples}"
        self.name += f"_p{self.explore_proba * 100:.0f}"
        self.name += f"_s{self.max_sigma * 100:.0f}"

    def _init_terminal_paths(self):
        self.terminal_paths = list(self.game.terminals.keys())
        random.shuffle(self.terminal_paths)

    def set_compare_agent(self, agent):
        self.compare_agent = agent

    def sample_trajectory(self, card1, card2) -> dict:
        """Perform random trajectory and returns dict of info."""
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            probs = self.get_policy(infoset)
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]
            history += action
        return visited

    def print_progress(self, epoch, t_long=600):
        time_left = None
        tot_epochs = self.epochs * self.deck_size * (self.deck_size - 1)

        if self.t0 is None:
            self.t0 = time()
            self.epoch_start = 0  # Track progress at the last reset
        else:
            time_elapsed = time() - self.t0

            # Check if 10 minutes have passed
            if time_elapsed > t_long:
                self.t0 = time()
                self.epoch_start = epoch
                time_elapsed = 0.001  # Prevent division by zero immediately after reset

            # Calculate speed based on progress since the last reset
            recent_progress = epoch - self.epoch_start
            if time_elapsed > 0:
                speed = recent_progress / time_elapsed
                remaining_epochs = tot_epochs - epoch
                if speed > 0:
                    time_left = remaining_epochs / speed

        p = epoch / tot_epochs
        out = f"{epoch:6})  {p:6.2%}  S = {self.entropy():.4f}   F = {self.best_card_fold_index():.4f}"

        if self.compare_agent is not None:
            ev_self, ev_other = evaluate_agents(self.game, self, self.compare_agent, n_games=self.n_games_compare)
            self.ev_list.append(ev_self)
            out += f"   EV: {ev_self:.2f}"

        if time_left is not None:
            out += f"   time left: {time_left / 60:.0f} min"
        print(out, end='\n')

    def plot_training_ev(self, label=""):
        plt.plot(self.ev_list, label=label)

    def get_baseline(self, actions, action_values, infoset) -> float:
        # Calculate the baseline (expected value) for the current policy
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * action_values[a] for a in actions)
        return baseline

    def _update_rule(self, actions, action_values, infoset):
        """ Direct Update: Update logits based on advantage. """
        baseline = self.get_baseline(actions, action_values, infoset)

        for action in actions:
            # Advantage-based update rule (NOT weighted by reach probability)
            advantage = action_values[action] - baseline
            update_step = self.lr * advantage
            self.update_logit(infoset, action, update_step)

        # Recalculate probabilities for this infoset
        self.softmax_update(infoset)

    def get_action_values(self, actions, history, card1, card2) -> dict:
        """ Calculate action values using rollouts. """
        player = len(history) % 2
        action_values = {}
        for action in actions:
            rewards = self.evaluate_action(history, action, card1, card2, self.rollout_samples)
            action_values[action] = rewards[player]
        return action_values

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

    def train(self, print_period=1000):
        """
        Training loop:
        1. Accumulate updates for all card combinations.
        2. Apply updates and center logits once per epoch.
        """
        print(f"\nTraining {self} ...")
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

    def get_next_path(self):
        """Cycle through the shuffled terminal paths."""
        path = self.terminal_paths[self._path_pointer]
        self._path_pointer = (self._path_pointer + 1) % len(self.terminal_paths)
        return path

    def sample_random_trajectory(self, card1, card2) -> Dict[Infoset, float]:
        trajectory = self.get_next_path()
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            actions = self.game.tree[history]
            action = trajectory[len(history)]
            reach_prob *= (1.0 / len(actions))
            history += action
        return visited

    def _get_posterior(self, my_card, history):
        """
        Re-uses your Bayesian inference logic to determine
        what the opponent is likely holding.
        """
        game = self.game
        num_cards = game.deck_size
        opponent_probs = np.zeros(num_cards)

        for card_opp in range(num_cards):
            if card_opp == my_card:
                continue

            reach_prob = 1.0
            temp_hist = ""
            for i, action in enumerate(history):
                acting_player = i % 2
                my_player_index = len(history) % 2
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
            post[my_card] = 0
            return post

    def evaluate_action(self, history, action, card1, card2, rollout_samples) -> tuple:
        """
        Overrides the standard evaluation. For each sample, it re-samples
        the opponent's card based on what their actions have revealed so far.
        """
        total_p1 = 0
        total_p2 = 0
        player_turn = len(history) % 2

        # Calculate posterior for the CURRENT player (the one we are evaluating)
        # and for the OPPONENT.
        post_p1 = self._get_posterior(card2, history)
        post_p2 = self._get_posterior(card1, history)

        for _ in range(rollout_samples):
            # Sample "realistic" hands for this rollout
            # If it's P1's turn to be evaluated, we keep card1 but sample a realistic card2
            if player_turn == 0:
                s_card1 = card1
                s_card2 = np.random.choice(range(self.deck_size), p=post_p1)
            else:
                s_card1 = np.random.choice(range(self.deck_size), p=post_p2)
                s_card2 = card2

            temp_h = history + action
            r1, r2 = self.rollout(temp_h, s_card1, s_card2)
            total_p1 += r1
            total_p2 += r2

        return total_p1 / rollout_samples, total_p2 / rollout_samples
