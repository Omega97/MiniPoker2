import random
import math
from typing import Dict
from dataclasses import dataclass
import os
import json
import ast
from pathlib import Path
from mini_poker.game import MiniPoker
from mini_poker.utils import clip
from mini_poker.paths import DATA_DIR


@dataclass
class Infoset:
    card: int
    branch: str

    def get_values(self) -> tuple:
        return self.card, self.branch

    def __repr__(self):
        return f'Infoset({self.card}, "{self.branch}")'

    def __iter__(self):
        yield from self.get_values()

    def __hash__(self):
        return hash(self.get_values())


def to_infoset(key_str: str) -> Infoset:
    """
    Helper to reconstruct Infoset from string key:
    [card, 'history']
    """
    card, history = ast.literal_eval(key_str)
    return Infoset(card, history)


class BaseAgent:
    """
    Generic tabular policy agent.

    Responsibilities:
    - store logits
    - convert logits -> softmax policy
    - sample actions
    """
    def __init__(self, game: MiniPoker, logit_bound=10.):
        self.game = game
        self.logit_bound = logit_bound
        self.deck_size = game.deck_size
        self.game_tree = game.tree
        self.logits = {}
        self.policy = {}
        self.name = None

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

    def rollout(self, history, card1, card2):
        """Returns (p1_reward, p2_reward)"""
        h = history
        while h not in self.game.terminals:
            player = len(h) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, h)
            action = self.get_action(infoset)
            h += action
        return self.game.get_reward(h, card1, card2)

    def evaluate_action(self, history, action, card1, card2, rollout_samples) -> tuple:
        """Evaluate action by averaging rollout rewards."""
        total_p1 = 0
        total_p2 = 0
        for _ in range(rollout_samples):
            temp_h = history + action
            r1, r2 = self.rollout(temp_h, card1, card2)
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

    def show_policy(self):
        """Print learned policies."""
        out = f"\nCOMPLETE LEARNED POLICY for N={self.game.game_power} (Stack={self.game.stack})\n\n"
        out += f"{'Card':<5} | {'History':<8} | {'Action Probabilities'}\n"
        out += "-" * 60 + '\n'
        histories = sorted(self.game.tree.keys(), key=lambda x: (len(x), x))
        for history in histories:
            for card in range(self.game.deck_size):
                infoset = Infoset(card, history)
                probs = self.get_policy(infoset)
                prob_str = "   ".join(f"{action} {prob:<4.0%}" for action, prob in probs.items())
                prob_str = prob_str.replace(f" {0:.0%}", " - ")
                display_h = history if history else "(root)"
                out += f" {card:<4} | {display_h:<8} |   {prob_str}\n"
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
                    infoset_entropy -= prob * math.log(prob)

            total_entropy += infoset_entropy

        return total_entropy / num_infosets

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

    def train(self):
        """Blanket training method"""
        pass
