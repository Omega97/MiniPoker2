# agents/crm_agent.py
import numpy as np
import random
import json
import os
from pathlib import Path
from typing import Dict
from collections import defaultdict
from mini_poker.paths import DATA_DIR
from mini_poker.game import to_infoset, Trajectory, Action
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import Infoset
from mini_poker.game import MiniPoker, State


class CRMAgent(BaseAgent):
    """
    Counterfactual Regret Minimization Agent (CFR).
    """
    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 explore_proba=0.1,
                 memory_period=100,
                 ):
        """
        :param game: Game instance
        :param logit_bound: logits are limited between - and + logit_bound
        :param epochs: number of training epochs
        :param explore_proba: random move proba during training
        """
        self.epochs = epochs
        self.cumulative_regrets = {}
        self.cumulative_policy = {}
        self.iteration_count = 0

        super().__init__(game, logit_bound, explore_proba=explore_proba, epochs=epochs, memory_period=memory_period)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"
        return bar

    def save(self, filepath: str):
        """Save agent including cumulative regrets, policy, and reward statistics."""
        data = {
            "logits": {str(list(k)): v for k, v in self.logits.items()},
            "policy": {str(list(k)): v for k, v in self.policy.items()},
            "cumulative_regrets": {str(list(k)): v for k, v in self.cumulative_regrets.items()},
            "cumulative_policy": {str(list(k)): v for k, v in self.cumulative_policy.items()},
            "average_reward": {
                str(list(infoset)): {action: val for action, val in actions.items()}
                for infoset, actions in self.average_reward.items()
            },
            "reward_counts": {
                str(list(infoset)): {action: count for action, count in actions.items()}
                for infoset, actions in self.reward_counts.items()
            },
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
        """Load agent including cumulative regrets, policy, and reward statistics."""
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

        if "average_reward" in data:
            self.average_reward = defaultdict(lambda: defaultdict(float))
            for infoset_str, actions in data["average_reward"].items():
                infoset = to_infoset(infoset_str)
                for action, value in actions.items():
                    self.average_reward[infoset][action] = float(value)

        if "reward_counts" in data:
            self.reward_counts = defaultdict(lambda: defaultdict(int))
            for infoset_str, actions in data["reward_counts"].items():
                infoset = to_infoset(infoset_str)
                for action, count in actions.items():
                    self.reward_counts[infoset][action] = int(count)

        print(f"\nAgent {self} loaded from {path}")

    # ======== Policy ========

    def _init_policy(self):
        """Initialize regrets and uniform policy for every infoset."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.cumulative_regrets[infoset] = {a: 0.0 for a in actions}
                self.cumulative_policy[infoset] = {a: 0.0 for a in actions}
                self.set_logits(infoset, {a: 0.0 for a in actions})
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def get_average_policy(self, infoset: Infoset) -> dict:
        """
        Get the average policy over all iterations.

        This is what CFR converges to, NOT the current policy.

        In CFR, the "current" strategy can be erratic.
        The average strategy over time is what eventually becomes unbeatable (Nash equilibrium).
        This method calculates that stable, long-term strategy.
        """
        actions = list(self.cumulative_policy[infoset].keys())
        total = sum(self.cumulative_policy[infoset].values())

        if total > 0:
            return {a: self.cumulative_policy[infoset][a] / total for a in actions}
        else:
            c = 1.0 / len(actions)
            return {a: c for a in actions}

    def get_action(self, infoset: Infoset):
        """Sample action according to the AVERAGE policy."""
        probs = self.get_average_policy(infoset)
        actions = list(probs.keys())
        p = list(probs.values())
        return np.random.choice(actions, p=p)

    # ======== CFR ========

    def regret_matching(self, infoset: Infoset) -> Dict[str, float]:
        """
        Convert cumulative regrets to policy using regret matching.

        This is the heart of the learning. It looks at which moves the agent "regrets"
        not taking and makes them more likely to happen in the next round.
        If a move was a total disaster, the agent stops doing it.

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
            c = 1.0 / len(actions)
            return {a: c for a in actions}

    def sample_trajectory_logic(self, state) -> Trajectory:
        """
        Sample trajectory by using the policy (or at random).
        The trajectory gets saved in the trajectories_cache.
        """
        trajectory = Trajectory(state)

        while not self.game.is_terminal(trajectory.state.branch):
            infoset = trajectory.get_current_player_infoset()
            probs = self.policy[infoset]
            actions = list(probs.keys())

            # Explicit exploration: Random action with probability explore_proba
            if random.random() < self.explore_proba:
                action = random.choice(actions)
            else:
                action = random.choices(actions, weights=list(probs.values()))[0]

            trajectory.perform_action(action, action_proba=probs[action])

        return trajectory

    def get_counterfactual_value(self, state: State, action: Action, player: int) -> float:
        """
        Calculate counterfactual value for taking a specific action.

        This is the expected value assuming we reach this infoset and take this action,
        then continue with current policy.
        """
        temp_state = state
        temp_state = temp_state.perform_action(action)

        # Complete the game with current policy
        while not self.game.is_terminal(temp_state.branch):
            curr_infoset = temp_state.get_current_player_infoset()
            probs = self.get_policy(curr_infoset)
            next_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            temp_state = temp_state.perform_action(next_action)

        rewards = self.game.get_reward(temp_state)

        return rewards[player]

    def new_policy_weight(self):
        """
        # Accumulate for averaging with CFR+ linear weighting
        # Weight recent policies more heavily for faster convergence
        """
        # return 1  # Constant
        return self.iteration_count  # Linear weighting (CFR+)

    def update_regrets(self, trajectory: Trajectory):
        """
        Update cumulative regrets for current player's infosets visited during the trajectory.
        """

        # We need the cards from the terminal state to reconstruct the game at each node
        card_p1 = trajectory.state.card_p1
        card_p2 = trajectory.state.card_p2

        # 1. Iterate through every situation (infoset) the agent encountered
        # 'rp0' and 'rp1' are the probabilities of reach for each player
        for infoset, (rp0, rp1) in trajectory.infoset_proba_pairs:
            player = infoset.get_current_player()
            actions = list(self.cumulative_regrets[infoset].keys())

            # COUNTERFACTUAL REACH: The probability that the OPPONENT played to reach this node.
            # If P0 is learning, we use P1's reach. If P1 is learning, we use P0's reach.
            cf_reach_prob = rp1 if player == 0 else rp0

            # Reconstruct the game state at this specific point in the past
            temp_state = State(card_p1, card_p2, branch=infoset.branch)

            # 2. Calculate values for every possible alternative action (the "What-if" rollouts)
            action_values = {}
            for action in actions:
                # This uses your rollout simulation to estimate the outcome of each move
                action_values[action] = self.get_counterfactual_value(temp_state, action, player)

            # 3. Calculate the Expected Value (EV) of our current policy at this node
            current_policy = self.get_policy(infoset)
            node_ev = sum(current_policy[a] * action_values[a] for a in actions)

            # 4. UPDATE CUMULATIVE REGRETS
            for action in actions:
                # Regret = (Value of action) - (Value of current strategy)
                regret = action_values[action] - node_ev

                # We weight the regret by the probability that the game actually reached here
                self.cumulative_regrets[infoset][action] += cf_reach_prob * regret

                # CFR+ Tweak: "Regret Floor"
                # If regret is negative, reset to 0 so the agent doesn't get stuck in the past.
                self.cumulative_regrets[infoset][action] = max(0.0, self.cumulative_regrets[infoset][action])

            # 5. UPDATE CURRENT POLICY (Regret Matching)
            # This adjusts the policy the agent will use in the NEXT training walk
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # 6. UPDATE AVERAGE POLICY (The target "Nash" strategy)
            # We weight this by the iteration count (CFR+ linear weighting)
            # and the player's own probability of reaching this node.
            weight = self.new_policy_weight()
            player_reach = rp0 if player == 0 else rp1

            for action in actions:
                self.cumulative_policy[infoset][action] += player_reach * new_policy[action] * weight

    def _exploration(self, card1, card2):
        """
        Sample trajectory
        """
        state = State(card1, card2, branch="")
        with self.train_context():
            trajectory = self.sample_trajectory_from_root(state)
        return trajectory

    def training_epoch(self):
        """
        One epoch of CFR training.

        Iterate through all card combinations and update regrets.
        """

        # Sample trajectories and update regrets
        for card1, card2 in self.game.iter_uniformly_over_hands():
            trajectory = self._exploration(card1, card2)
            self.update_regrets(trajectory)
            self.iteration_count += 1

        # Clear list of trajectories
        self.clear_trajectories_cache()
