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

    Key fixes implemented:
    1. Track BOTH players' reach probabilities separately
    2. Use OPPONENT'S reach probability for counterfactual updates
    3. Only update regrets for current player's infosets
    4. Use time-averaged policy for decision making
    5. Add explicit exploration for better early training
    6. CFR+ linear weighting for faster convergence
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 n_games_compare=10_000,
                 explore_proba=0.1,  # Explicit exploration parameter
                 ):
        self.epochs = epochs
        self.explore_proba = explore_proba
        self.cumulative_regrets = {}
        self.cumulative_policy = {}
        self.iteration_count = 0

        super().__init__(game, logit_bound, epochs=epochs, n_games_compare=n_games_compare)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

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
        """
        Convert cumulative regrets to policy using regret matching.

        π(I, a) = R+(I, a) / Σ R+(I, b)
        where R+ = max(0, R)

        If all regrets are ≤ 0, use uniform policy.
        """
        actions = list(self.cumulative_regrets[infoset].keys())
        positive_regrets = {a: max(0, self.cumulative_regrets[infoset][a]) for a in actions}
        sum_positive = sum(positive_regrets.values())

        if sum_positive > 0:
            return {a: r / sum_positive for a, r in positive_regrets.items()}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_average_policy(self, infoset: Infoset) -> dict:
        """
        Get the average policy over all iterations.

        CRITICAL: This is what CFR converges to, NOT the current policy.
        """
        actions = list(self.cumulative_policy[infoset].keys())
        total = sum(self.cumulative_policy[infoset].values())

        if total > 0:
            return {a: self.cumulative_policy[infoset][a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_action(self, infoset: Infoset):
        """
        CRITICAL: Use time-averaged policy for decisions, NOT current policy.

        CFR's theoretical guarantees apply to the average policy, not the
        instantaneous regret-matched policy.
        """
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
        Sample trajectory and track BOTH players' reach probabilities separately.

        Returns: (visited_infosets, reach_prob_p0, reach_prob_p1)

        CRITICAL FIX: We need separate reach probabilities for each player
        to compute proper counterfactual values.
        """
        visited = {}
        history = ""
        reach_prob_p0 = 1.0  # Player 0's reach probability
        reach_prob_p1 = 1.0  # Player 1's reach probability

        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)

            visited[infoset] = (reach_prob_p0, reach_prob_p1)

            probs = self.get_policy(infoset)  # Use current policy for sampling
            actions = list(probs.keys())

            # EXPLICIT EXPLORATION: Random action with probability explore_proba
            if random.random() < self.explore_proba:
                action = random.choice(actions)
            else:
                action = random.choices(actions, weights=list(probs.values()))[0]

            # Update reach probabilities based on whose turn it is
            if player == 0:
                reach_prob_p0 *= probs[action]
            else:
                reach_prob_p1 *= probs[action]

            history += action

        return visited, reach_prob_p0, reach_prob_p1

    def get_counterfactual_value(self, state: State, action: str, player: int) -> float:
        """
        Calculate counterfactual value for taking a specific action.

        This is the expected value assuming we reach this infoset and take this action,
        then continue with current policy.
        """
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

        CRITICAL FIXES:
        1. Use OPPONENT'S reach probability (not your own)
        2. Only update regrets for the current player at each infoset
        3. Accumulate policy for averaging with CFR+ linear weighting
        """
        visited, reach_prob_p0, reach_prob_p1 = self.sample_trajectory(card1, card2)

        for infoset, (rp0, rp1) in visited.items():
            history = infoset.branch
            player = len(history) % 2

            # CRITICAL: Use OPPONENT'S reach probability for counterfactual updates
            # Player 0 uses player 1's reach, Player 1 uses player 0's reach
            if player == 0:
                cf_reach_prob = rp1  # Player 0's counterfactual reach = P1's actual reach
            else:
                cf_reach_prob = rp0  # Player 1's counterfactual reach = P0's actual reach

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

            # Update cumulative regrets
            for action in actions:
                regret = action_values[action] - policy_value
                self.cumulative_regrets[infoset][action] += cf_reach_prob * regret

            # Update current policy using regret matching
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # CRITICAL: Accumulate for averaging with CFR+ linear weighting
            # Weight recent policies more heavily for faster convergence
            weight = self.iteration_count  # Linear weighting (CFR+)
            for action in new_policy:
                self.cumulative_policy[infoset][action] += new_policy[action] * weight

    def training_epoch(self):
        """
        One epoch of CFR training.

        Iterate through all card combinations and update regrets.
        """
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
            "explore_proba": self.explore_proba,
            "game_params": {
                "game_power": self.game.game_power,
                "deck_size": self.deck_size,
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
        self.explore_proba = data.get("explore_proba", 0.1)

        print(f"\nAgent {self} loaded from {path}")
