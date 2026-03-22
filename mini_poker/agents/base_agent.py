import random
import math
from typing import Dict
import os
import json
from pathlib import Path
from time import time
import matplotlib.pyplot as plt
import numpy as np
from mini_poker.game import MiniPoker, Infoset, to_infoset, State
from mini_poker.utils import clip
from mini_poker.paths import DATA_DIR
from mini_poker.training.evaluation import evaluate_agents
from mini_poker.utils import print_colored_status, COLORS


class BaseAgent:
    """
    Generic tabular policy agent.

    Responsibilities:
    - store logits
    - convert logits -> softmax policy
    - sample actions
    """
    def __init__(self, game: MiniPoker, logit_bound=10., epochs=None, n_games_compare=10_000):
        self.game = game
        self.epochs = epochs
        self.logit_bound = logit_bound
        self.n_games_compare = n_games_compare
        self.deck_size = game.deck_size
        self.game_tree = game.tree
        self.logits = {}
        self.policy = {}
        self.name = None
        self.compare_agent = None
        self.epoch = None
        self.ev_list = None
        self.times = None

        self._init_policy()
        self._init_name()

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"

    def __str__(self):
        return self.name

    def get_logit(self, infoset: Infoset, action: str) -> float:
        return self.logits[infoset][action]

    def set_logit(self, infoset: Infoset, action: str, logit: float):
        value = clip(logit, -self.logit_bound, self.logit_bound)
        self.logits[infoset][action] = value

    def update_logit(self, infoset: Infoset, action: str, update_step: float):
        """Add update_step and clip for stability."""
        value = self.get_logit(infoset, action) + update_step
        self.set_logit(infoset, action, value)

    def center_logits(self, infoset: Infoset) -> None:
        """Force sum of logits to 0."""
        logits_dict = self.logits[infoset]
        actions = list(logits_dict.keys())

        # Compute mean of current logits
        mean_logit = sum(logits_dict.values()) / len(actions)

        # Subtract mean from each logit (and re-apply clipping)
        for action in actions:
            centered = logits_dict[action] - mean_logit
            self.set_logit(infoset, action, centered)  # set_logit already clips

    def get_logits(self, infoset: Infoset) -> Dict[str, float]:
        return self.logits[infoset]

    def set_logits(self, infoset: Infoset, logits: Dict[str, float]):
        """Set logits one by one (with clipping)."""
        if infoset not in self.logits:
            self.logits[infoset] = dict()
        for action, logit in logits.items():
            self.set_logit(infoset, action, logit)

    def get_policy(self, infoset: Infoset) -> Dict[str, float]:
        return self.policy[infoset]

    def set_policy(self, infoset: Infoset, proba: Dict[str, float]):
        self.policy[infoset] = proba

    def _init_policy(self):
        """Initialize logits and uniform policy for every infoset."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.set_logits(infoset, {a: 0.0 for a in actions})
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def get_action(self, infoset: Infoset):
        """Sample action according to current policy."""
        probs = self.get_policy(infoset)
        r = random.random()
        cumulative = 0.0
        for action in probs:
            cumulative += probs[action]
            if r < cumulative:
                return action
        return list(probs.keys())[-1]

    def softmax_update(self, infoset: Infoset):
        """Convert logits to policy probabilities."""
        raw_logits = self.get_logits(infoset)
        max_logit = max(raw_logits.values())
        exps = {a: math.exp(v - max_logit) for a, v in raw_logits.items()}
        Z = sum(exps.values())
        proba = {a: exps[a] / Z for a in raw_logits}
        self.set_policy(infoset, proba)

    def rollout(self, state: State):
        """Returns (p1_reward, p2_reward)"""
        while state.branch not in self.game.terminals:
            player = len(state.branch) % 2
            card = state.card_p1 if player == 0 else state.card_p2
            infoset = Infoset(card, state.branch)
            action = self.get_action(infoset)
            state.branch += action
        return self.game.get_reward(state)

    def evaluate_action(self, state: State, action, rollout_samples) -> tuple:
        """Evaluate action by averaging rollout rewards."""
        total_p1 = 0
        total_p2 = 0
        for _ in range(rollout_samples):
            temp_state = state.copy()
            temp_state.branch += action
            r1, r2 = self.rollout(temp_state)
            total_p1 += r1
            total_p2 += r2
        value_p1 = total_p1 / rollout_samples
        value_p2 = total_p2 / rollout_samples
        return value_p1, value_p2

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
            probs = self.policy[infoset]
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]
            history += action
        return visited

    def show_policy(self, logit_mode=False):
        """
        Print learned policies or raw logits.
        :param logit_mode: If True, displays raw logit values instead of percentages.
        """
        mode_title = "LOGITS" if logit_mode else "ACTION PROBABILITIES"
        out = f"\nCOMPLETE LEARNED {mode_title} for N={self.game.game_power} (Stack={self.game.stack})\n\n"
        out += f"{'Card':<5} | {'History':<8} | {mode_title}\n"
        out += "-" * 70 + '\n'

        # Sort histories by length then alphabetical order
        histories = sorted(self.game.tree.keys(), key=lambda x: (len(x), x))

        for history in histories:
            for card in range(self.game.deck_size):
                infoset = Infoset(card, history)

                if logit_mode:
                    # Retrieve raw logit values
                    data = self.get_logits(infoset)
                    data_str = "   ".join(f"{action} {val:>6.2f}" for action, val in data.items())
                else:
                    # Retrieve softmax policy probabilities
                    probs = self.get_policy(infoset)
                    data_str = "   ".join(f"{action} {prob:<4.0%}" for action, prob in probs.items())
                    data_str = data_str.replace(f" {0:.0%}", " - ")

                display_h = history if history else "(root)"
                out += f" {card:<4} | {display_h:<8} |   {data_str}\n"
            out += '\n'
        return out

    def save(self, filepath: str):
        """
        Saves the agent's logits and policy to a JSON file.
        """
        # Convert dictionary keys (Infoset objects) to strings for JSON serialization
        data = {
            "logits": {str(list(k)): v for k, v in self.logits.items()},
            "policy": {str(list(k)): v for k, v in self.policy.items()},
            "game_params": {
                "game_power": self.game.game_power,
                "deck_size": self.game.deck_size,
                "stack": self.game.stack
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str = None):
        """
        Loads the agent's logits and policy from a JSON file.
        Continues silently if the file does not exist.
        """
        if filepath:
            path = Path(filepath)
        else:
            path = Path(DATA_DIR / f"{self}.json")

        with open(path, 'r') as f:
            data = json.load(f)

        # Using your existing conversion logic
        self.logits = {to_infoset(k): v for k, v in data["logits"].items()}
        self.policy = {to_infoset(k): v for k, v in data["policy"].items()}
        print(f"\nAgent {self} loaded from {path}")

    def entropy(self) -> float:
        """
        Calculates the average Shannon entropy across all
        information sets in the current policy.
        """
        if not self.policy:
            return 0.0

        total_entropy = 0.0
        num_infosets = len(self.policy)

        for infoset, probs in self.policy.items():
            infoset_entropy = 0.0
            for prob in probs.values():
                # Entropy is 0 if probability is 0; use a small epsilon or check > 0
                if prob > 0:
                    infoset_entropy -= prob * math.log2(prob)

            total_entropy += infoset_entropy

        return total_entropy / num_infosets

    def terminal_entropy(self) -> float:
        """
        Calculates the average Shannon entropy across only the
        information sets that lead directly to terminal states.
        """
        if not self.policy:
            return 0.0

        total_entropy = 0.0
        terminal_infoset_count = 0

        for infoset, probs in self.policy.items():
            # Check if all possible actions from this history result in a terminal state
            actions = self.game_tree.get(infoset.branch, [])
            if not actions:
                continue

            is_terminal_decision = all(
                (infoset.branch + a) in self.game.terminals
                for a in actions
            )

            if is_terminal_decision:
                infoset_entropy = 0.0
                for prob in probs.values():
                    if prob > 0:
                        infoset_entropy -= prob * math.log2(prob)

                total_entropy += infoset_entropy
                terminal_infoset_count += 1

        if terminal_infoset_count == 0:
            return 0.0

        return total_entropy / terminal_infoset_count

    def best_card_fold_index(self) -> float:
        """
        Returns a metric where:
        - 1.0: Agent is folding the best card at the same rate as a random agent.
        - 0.0: Agent never folds the best card.
        - > 1.0: Agent is actively 'punishing' the best card (worse than random).
        """
        if not self.policy:
            return 0.0

        best_card = self.deck_size - 1
        total_weighted_score = 0.0
        nodes_counted = 0
        fold_action_keys = {'F', 'f'}

        for infoset, probs in self.policy.items():
            if infoset.card == best_card:
                # Check if 'fold' is even an option at this node
                available_actions = list(probs.keys())
                num_actions = len(available_actions)

                # Find the fold action
                fold_prob = 0.0
                has_fold = False
                for action in available_actions:
                    if action in fold_action_keys:
                        fold_prob = probs[action]
                        has_fold = True
                        break

                if has_fold:
                    # Weight by number of actions so (1/N) * N = 1.0
                    total_weighted_score += (fold_prob * num_actions)
                    nodes_counted += 1

        if nodes_counted == 0:
            return -1.0

        return total_weighted_score / nodes_counted

    def inherit_from(self, other_agent):
        self.logits = other_agent.logits.copy()
        self.policy = other_agent.policy.copy()

    def sanity_check(self, rel_tol=1e-5):
        """
        Counts decision-points (infosets) where all possible actions
        have hit the logit boundary (+/- logit_bound).
        """
        saturated_count = 0
        total_infosets = len(self.logits)

        for infoset, action_logits in self.logits.items():
            # Check if every logit in this infoset is at the boundary
            # We use math.isclose to handle potential floating point precision issues
            is_saturated = all(
                math.isclose(abs(logit), self.logit_bound, rel_tol=rel_tol)
                for logit in action_logits.values()
            )

            if is_saturated:
                saturated_count += 1

        print(f"Sanity Check: {saturated_count}/{total_infosets} infosets are fully saturated.")
        return saturated_count

    def set_compare_agent(self, agent):
        """Set agent to compare with during training."""
        self.compare_agent = agent

    def training(self):
        """Blanket training method"""
        print(f"\nTraining {self} ...")
        self.ev_list = []
        self.times = [time()]
        print(self.get_progress_bar())
        for self.epoch in range(self.epochs):
            self.training_epoch()
            if self.compare_agent is not None:
                ev_self, ev_other = evaluate_agents(self.game, self, self.compare_agent, n_games=self.n_games_compare)
                self.ev_list.append(ev_self)
            self.times.append(time())
            print(self.get_progress_bar())

    def get_progress_bar(self, epsilon=1e-6):
        bar = ""

        # Epochs
        if self.epoch is not None:
            bar += f"{self.epoch+1:3})"
            bar += f"  {(self.epoch+1) / self.epochs:7.2%}"
        else:
            bar += f"{0:3})"
            bar += f"  {0:7.2%}"

        # EVs
        if self.ev_list is not None and len(self.ev_list):
            ev = self.ev_list[-1]
            bar += print_colored_status(ev, text=f"  {ev:+6.2f}", red=COLORS['white'])
        else:
            bar += f"    --- "

        # Time
        if self.times is not None and len(self.times) >= 2:
            time_elapsed = self.times[-1] - self.times[-2]
            speed = 1 / (time_elapsed + epsilon)
            epochs_left = self.epochs - self.epoch
            time_left = epochs_left / speed
            if time_left > 60:
                bar += f"   eta: {time_left//60:3.0f} min"
            else:
                bar += f"   eta: {time_left:3.0f} s  "
        else:
            bar += f"   eta: ???    "

        return bar

    def plot_training_ev(self, label="", max_window_size=50):
        if not self.ev_list:
            return

        y = np.array(self.ev_list)
        n = len(y)
        y_smooth = np.zeros(n)

        for i in range(n):
            # 1. Determine dynamic window size
            # (Growing window: small at start, large at end)
            win_size = max(1, int(((i + 1) / n) * max_window_size))

            # 2. Slice the data for the current window
            start_idx = max(0, i - win_size + 1)
            window_data = y[start_idx: i + 1]
            current_len = len(window_data)

            if current_len == 1:
                y_smooth[i] = window_data[0]
                continue

            # 3. Create Linear Kernel Weights: y = 2x
            # x is normalized distance within the window [0, 1]
            # We want the most recent point (index -1) to have the highest weight
            weights = np.linspace(0.1, 1.0, current_len)

            # 4. Apply Weighted Average
            y_smooth[i] = np.average(window_data, weights=weights)

        plt.plot(y_smooth, label=label)

    def training_epoch(self):
        pass
