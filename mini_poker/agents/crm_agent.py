# agents\crm_agent.py
from itertools import permutations
import random
import json
from pathlib import Path
from mini_poker.paths import DATA_DIR
from mini_poker.game import to_infoset
import os
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import Infoset
from mini_poker.game import MiniPoker, State


class CRMAgent(BaseAgent):
    """
    Counterfactual Regret Minimization Agent (CFR).

    Key corrections:
    1. Proper counterfactual reach probability (opponent only)
    2. Average policy for decision making
    3. Only update regrets for current player's infosets
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 n_games_compare=10_000,
                 ):
        self.epochs = epochs
        self.cumulative_regrets = {}
        self.cumulative_policy = {}
        self.iteration_count = 0

        super().__init__(game, logit_bound, epochs=epochs, n_games_compare=n_games_compare)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"

    def _init_policy(self):
        """Initialize regrets and uniform policy for every infoset."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.cumulative_regrets[infoset] = {a: 0.0 for a in actions}
                self.cumulative_policy[infoset] = {a: 0.0 for a in actions}
                self.set_logits(infoset, {a: 0.0 for a in actions})
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def regret_matching(self, infoset: Infoset) -> dict:
        """Convert cumulative regrets to policy using regret matching."""
        actions = list(self.cumulative_regrets[infoset].keys())
        positive_regrets = {a: max(0, self.cumulative_regrets[infoset][a]) for a in actions}
        sum_positive = sum(positive_regrets.values())

        if sum_positive > 0:
            return {a: r / sum_positive for a, r in positive_regrets.items()}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_average_policy(self, infoset: Infoset) -> dict:
        """Get the average policy over all iterations."""
        if self.iteration_count == 0:
            return self.get_policy(infoset)

        actions = list(self.cumulative_policy[infoset].keys())
        total = sum(self.cumulative_policy[infoset].values())

        if total > 0:
            return {a: self.cumulative_policy[infoset][a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_action(self, infoset: Infoset):
        """Sample action from AVERAGE policy (not current policy)."""
        probs = self.get_average_policy(infoset)
        r = random.random()
        cumulative = 0.0
        for action in probs:
            cumulative += probs[action]
            if r < cumulative:
                return action
        return list(probs.keys())[-1]

    def sample_trajectory(self, card1, card2) -> tuple:
        """
        Sample trajectory and return (visited_infosets, counterfactual_reach_probs).
        Counterfactual reach prob = probability opponent reaches this infoset.
        """
        visited = {}
        cf_reach_probs = {}  # Counterfactual reach probabilities
        history = ""
        reach_prob = 1.0
        cf_reach_prob = 1.0  # Only opponent's actions

        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)

            visited[infoset] = reach_prob
            cf_reach_probs[infoset] = cf_reach_prob  # Store counterfactual reach

            probs = self.get_policy(infoset)  # Use current policy for sampling
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]

            # Only multiply opponent's actions for counterfactual reach
            if player != 0:  # If opponent's turn (for player 0's perspective)
                cf_reach_prob *= probs[action]

            history += action

        return visited, cf_reach_probs

    def get_counterfactual_value(self, state: State, action: str, player: int) -> float:
        """Calculate counterfactual value for taking a specific action."""
        temp_state = state.copy()
        temp_state.branch += action

        # Complete the game with current policy
        while temp_state.branch not in self.game.terminals:
            curr_player = len(temp_state.branch) % 2
            curr_card = temp_state.card_p1 if curr_player == 0 else temp_state.card_p2
            curr_infoset = Infoset(curr_card, temp_state.branch)

            probs = self.get_policy(curr_infoset)
            next_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            temp_state.branch += next_action

        rewards = self.game.get_reward(temp_state)
        return rewards[player]

    def update_regrets(self, card1, card2):
        """
        Update cumulative regrets for current player's infosets only.
        Use counterfactual reach probabilities.
        """
        visited, cf_reach_probs = self.sample_trajectory(card1, card2)

        for infoset, cf_reach_prob in cf_reach_probs.items():
            history = infoset.branch
            player = len(history) % 2

            # Create state for this infoset
            state = State(card1, card2, branch=history)
            actions = self.game.tree[history]

            # Get counterfactual values for each action
            action_values = {}
            for action in actions:
                action_values[action] = self.get_counterfactual_value(state, action, player)

            # Get expected value of current policy
            probs = self.get_policy(infoset)
            policy_value = sum(probs[a] * action_values[a] for a in actions)

            # Update cumulative regrets (only for current player)
            for action in actions:
                regret = action_values[action] - policy_value
                self.cumulative_regrets[infoset][action] += cf_reach_prob * regret

            # Update current policy using regret matching
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # Accumulate policy for averaging
            for action in actions:
                self.cumulative_policy[infoset][action] += new_policy[action]

    def training_epoch(self):
        """One epoch of CFR training."""
        self.iteration_count += 1

        for card1, card2 in permutations(range(self.deck_size), 2):
            self.update_regrets(card1, card2)

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"
        return bar

    def save(self, filepath: str):
        """Save agent including cumulative regrets and policy."""
        data = {
            "logits": {str(list(k)): v for k, v in self.logits.items()},
            "policy": {str(list(k)): v for k, v in self.policy.items()},
            "cumulative_regrets": {str(list(k)): v for k, v in self.cumulative_regrets.items()},
            "cumulative_policy": {str(list(k)): v for k, v in self.cumulative_policy.items()},
            "iteration_count": self.iteration_count,
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
        """Load agent including cumulative regrets and policy."""
        if filepath:
            path = Path(filepath)
        else:
            path = Path(DATA_DIR / f"{self}.json")

        with open(path, 'r') as f:
            data = json.load(f)

        self.logits = {to_infoset(k): v for k, v in data["logits"].items()}
        self.policy = {to_infoset(k): v for k, v in data["policy"].items()}
        self.cumulative_regrets = {to_infoset(k): v for k, v in data["cumulative_regrets"].items()}
        self.cumulative_policy = {to_infoset(k): v for k, v in data["cumulative_policy"].items()}
        self.iteration_count = data.get("iteration_count", 0)

        print(f"\nAgent {self} loaded from {path}")
