# agents/cfr_agent.py
import numpy as np
import random
import json
import os
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
from mini_poker.paths import DATA_DIR
from mini_poker.game import to_infoset, Trajectory, Action, MiniPoker, State
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import Infoset


class CFRAgent(BaseAgent):
    """
    Counterfactual Regret Minimization Agent (CFR+ with DCFR discounting).

    Key features:
    - CFR+ regret flooring (negative regrets reset to 0)
    - DCFR discounting for faster convergence
    - Alternating player updates for stability
    - Separate training vs evaluation policies
    - Online CFR search during inference for fine-tuned decisions
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 memory_period=100,
                 dcfr_alpha=1.5,      # DCFR discount parameter (1.5 is standard)
                 dcfr_beta=1.0,       # DCFR discount parameter (1.0 is standard)
                 search_iterations=1000,  # Online CFR search iterations during inference
                 search_enabled=False,   # Toggle for online search
                 ):
        """
        :param game: Game instance
        :param logit_bound: Logits are limited between - and + logit_bound
        :param epochs: Number of training epochs
        :param dcfr_alpha: DCFR discount parameter (1.5 is standard)
        :param dcfr_beta: DCFR discount parameter (1.0 is standard)
        :param search_iterations: Number of CFR iterations during inference search
        :param search_enabled: Whether to use online CFR search during inference
        """
        self.epochs = epochs
        self.cumulative_regrets = {}
        self.cumulative_policy = {}
        self.iteration_count = 0
        self.current_player_update = 0  # Alternates between 0 and 1
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.search_iterations = search_iterations
        self.search_enabled = search_enabled

        # CFR doesn't need exploration during training
        super().__init__(game, logit_bound, explore_proba=0.0, epochs=epochs, memory_period=memory_period)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        if self.search_enabled:
            self.name += f"_s{self.search_iterations}"

    def regret_completeness_index(self) -> float:
        """
        Calculates the percentage of available actions in the game tree
        that have been visited/updated at least once.
        """
        total_actions = 0
        visited_actions = 0

        for infoset, actions in self.cumulative_policy.items():
            for action, weight in actions.items():
                total_actions += 1
                if weight > 0:
                    visited_actions += 1

        if total_actions == 0:
            return 0.0

        return visited_actions / total_actions

    def average_cumulative_regret(self) -> float:
        """Average magnitude of cumulative regrets across all infosets."""
        total = 0.0
        count = 0
        for infoset, regrets in self.cumulative_regrets.items():
            for r in regrets.values():
                total += abs(r)
                count += 1
        return total / count if count > 0 else 0.0

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   F = {self.best_card_fold_index():.3f}"
        bar += f"   r = {self.regret_completeness_index():7.2%}"
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   |R| = {self.average_cumulative_regret():6.2f}"
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
            "dcfr_params": {
                "alpha": self.dcfr_alpha,
                "beta": self.dcfr_beta
            },
            "search_params": {
                "iterations": self.search_iterations,
                "enabled": self.search_enabled
            },
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

        if "dcfr_params" in data:
            self.dcfr_alpha = data["dcfr_params"].get("alpha", 1.5)
            self.dcfr_beta = data["dcfr_params"].get("beta", 1.0)

        if "search_params" in data:
            self.search_iterations = data["search_params"].get("iterations", 100)
            self.search_enabled = data["search_params"].get("enabled", False)

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

        This is what CFR converges to (Nash equilibrium), NOT the current policy.
        Use this for evaluation/exploitation, not during training.
        """
        actions = list(self.cumulative_policy[infoset].keys())
        total = sum(self.cumulative_policy[infoset].values())

        if total > 1e-10:
            return {a: self.cumulative_policy[infoset][a] / total for a in actions}
        else:
            c = 1.0 / len(actions)
            return {a: c for a in actions}

    def get_training_action(self, infoset: Infoset):
        """
        Sample action according to CURRENT policy.

        Use this for TRAINING only (trajectory sampling during CFR iterations).
        The current policy drives regret updates.
        """
        probs = self.get_policy(infoset)  # Current regret-matched policy
        actions = list(probs.keys())
        p = list(probs.values())
        return np.random.choice(actions, p=p)

    def get_eval_action(self, infoset: Infoset, use_search: bool = None):
        """
        Explicit evaluation method with optional search.

        :param infoset: Current infoset
        :param use_search: If True, perform online CFR search before deciding
        """
        return self.get_action(infoset, use_search=use_search)

    # ======== CFR ========

    def regret_matching(self, infoset: Infoset) -> Dict[str, float]:
        """
        Convert cumulative regrets to policy using regret matching.

        π(I, a) = R+(I, a) / Σ R+(I, b)
        where R+ = max(0, R)

        If all regrets are ≤ 0, use uniform policy.
        """
        actions = list(self.cumulative_regrets[infoset].keys())
        positive_regrets = {a: max(0, self.cumulative_regrets[infoset][a]) for a in actions}
        sum_positive = sum(positive_regrets.values())

        if sum_positive > 1e-10:
            return {a: r / sum_positive for a, r in positive_regrets.items()}
        else:
            c = 1.0 / len(actions)
            return {a: c for a in actions}

    def sample_trajectory_logic(self, state) -> Trajectory:
        """
        Sample trajectory using CURRENT policy (for training).

        CFR converges by iterating through all game states systematically.
        Random exploration adds variance to regret estimates without benefit.
        """
        trajectory = Trajectory(state)

        while not self.game.is_terminal(trajectory.state.branch):
            infoset = trajectory.get_current_player_infoset()

            # Use CURRENT policy for training (not average!)
            probs = self.get_policy(infoset)
            actions = list(probs.keys())
            action = random.choices(actions, weights=list(probs.values()))[0]
            trajectory.perform_action(action, action_proba=probs[action])

        return trajectory

    def get_counterfactual_value(self, state: State, action: Action, player: int, n_samples: int = 1) -> float:
        """
        Calculate counterfactual value for taking a specific action.

        Optional multi-sampling reduces variance in regret estimates.
        """
        total_value = 0.0

        for _ in range(n_samples):
            temp_state = state.perform_action(action)

            while not self.game.is_terminal(temp_state.branch):
                curr_infoset = temp_state.get_current_player_infoset()
                probs = self.get_policy(curr_infoset)
                next_action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
                temp_state = temp_state.perform_action(next_action)

            rewards = self.game.get_reward(temp_state)
            total_value += rewards[player]

        return total_value / n_samples

    def new_policy_weight(self) -> float:
        """
        CFR+ linear weighting: weight = iteration_count + 1

        This gives more weight to recent policies for faster convergence.
        """
        return max(1, self.iteration_count + 1)

    def apply_dcfr_discount(self, infoset: Infoset):
        """
        Apply DCFR (Discounted CFR) discounting to cumulative regrets.

        DCFR improves convergence by discounting older regret values.
        Standard parameters: alpha=1.5, beta=1.0
        """
        if self.iteration_count > 0:
            # Discount cumulative regrets
            regret_discount = (self.iteration_count ** self.dcfr_alpha) / \
                              ((self.iteration_count + 1) ** self.dcfr_alpha)

            # Discount cumulative policy (optional, beta parameter)
            policy_discount = (self.iteration_count ** self.dcfr_beta) / \
                              ((self.iteration_count + 1) ** self.dcfr_beta)

            for action in self.cumulative_regrets[infoset]:
                self.cumulative_regrets[infoset][action] *= regret_discount
                self.cumulative_policy[infoset][action] *= policy_discount

    def update_regrets(self, trajectory: Trajectory, player_to_update: Optional[int] = None):
        """
        Update cumulative regrets for specified player's infosets.

        :param trajectory: The sampled trajectory
        :param player_to_update: Which player to update (0, 1, or None for both)
        """
        card_p1 = trajectory.state.card_p1
        card_p2 = trajectory.state.card_p2

        for infoset, (rp0, rp1) in trajectory.infoset_proba_pairs:
            player = infoset.get_current_player()

            # Skip if we're only updating one player and this isn't them
            if player_to_update is not None and player != player_to_update:
                continue

            actions = list(self.cumulative_regrets[infoset].keys())

            # COUNTERFACTUAL REACH: Probability that the OPPONENT played to reach this node
            cf_reach_prob = rp1 if player == 0 else rp0

            # Reconstruct the game state at this specific point
            temp_state = State(card_p1, card_p2, branch=infoset.branch)

            # Apply DCFR discounting before updates
            self.apply_dcfr_discount(infoset)

            # Calculate values for every possible alternative action
            action_values = {}
            for action in actions:
                action_values[action] = self.get_counterfactual_value(temp_state, action, player)

            # Calculate the Expected Value (EV) of our current policy at this node
            current_policy = self.get_policy(infoset)
            node_ev = sum(current_policy[a] * action_values[a] for a in actions)

            # UPDATE CUMULATIVE REGRETS
            for action in actions:
                # Regret = (Value of action) - (Value of current strategy)
                regret = action_values[action] - node_ev

                # Weight the regret by counterfactual reach probability
                self.cumulative_regrets[infoset][action] += cf_reach_prob * regret

                # CFR+ Regret Floor: If regret is negative, reset to 0
                self.cumulative_regrets[infoset][action] = max(0.0, self.cumulative_regrets[infoset][action])

            # UPDATE CURRENT POLICY (Regret Matching)
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # UPDATE AVERAGE POLICY (The target "Nash" strategy)
            weight = self.new_policy_weight()
            player_reach = rp0 if player == 0 else rp1

            for action in actions:
                self.cumulative_policy[infoset][action] += player_reach * new_policy[action] * weight

    def _exploration(self, card1, card2):
        """Sample trajectory from given cards."""
        state = State(card1, card2, branch="")
        with self.train_context():
            trajectory = self.sample_trajectory_from_root(state)
        return trajectory

    def training_epoch(self):
        """
        One epoch of CFR training with alternating player updates.

        Alternating which player's regrets are updated each epoch reduces
        correlation between players' regret signals and improves convergence.
        """
        # Alternate which player's regrets are updated
        self.current_player_update = self.epoch % 2

        # Sample trajectories and update regrets
        for card1, card2 in self.game.iter_uniformly_over_hands():
            trajectory = self._exploration(card1, card2)
            self.update_regrets(trajectory, player_to_update=self.current_player_update)
            self.iteration_count += 1

        # Clear list of trajectories
        self.clear_trajectories_cache()

    # ======== Online CFR Search (Inference-Time Fine-Tuning) ========

    def compute_posterior(self, infoset: Infoset) -> np.ndarray:
        """
        Compute posterior distribution over opponent's possible cards
        given the current infoset and the trained average policy.

        Uses Bayesian inference: P(opponent_card | history) ∝ P(history | opponent_card) * P(opponent_card)

        :param infoset: Current infoset (our card + history)
        :return: Array of probabilities for each opponent card
        """
        num_cards = self.game.deck_size
        opponent_probs = np.zeros(num_cards)

        # Prior: uniform over all cards except our own
        for card_opp in range(num_cards):
            if card_opp == infoset.card:
                opponent_probs[card_opp] = 0.0
                continue

            # Likelihood: probability opponent played the observed history with this card
            reach_prob = 1.0
            temp_hist = ""

            for i, action in enumerate(infoset.branch):
                acting_player = i % 2
                my_player_index = len(infoset.branch) % 2

                # If it was the OPPONENT's turn
                if acting_player != my_player_index:
                    prev_infoset = Infoset(card_opp, temp_hist)
                    if prev_infoset in self.cumulative_policy:
                        probs = self.get_average_policy(prev_infoset)
                        reach_prob *= probs.get(action, 0.0)
                    else:
                        # Unvisited infoset: assume uniform
                        if prev_infoset in self.game_tree:
                            actions = self.game_tree[prev_infoset.branch]
                            reach_prob *= 1.0 / len(actions) if actions else 1.0
                        else:
                            reach_prob *= 0.0

                temp_hist += action

            opponent_probs[card_opp] = reach_prob

        # Normalize to get posterior
        total_weight = np.sum(opponent_probs)
        if total_weight > 1e-10:
            posterior = opponent_probs / total_weight
        else:
            # Fallback: uniform over valid cards
            posterior = np.ones(num_cards) / (num_cards - 1)
            posterior[infoset.card] = 0.0

        return posterior

    # ======== Online CFR Search (Inference-Time Fine-Tuning) ========

    def online_search(self, infoset: Infoset, n_iterations: int = None) -> Dict[str, float]:
        """
        Perform online CFR search from the current infoset to fine-tune the policy.

        This runs additional CFR iterations focused on the current game state subtree,
        using the trained regrets/policy as initialization and sampling opponent cards
        from the posterior distribution.

        CRITICAL: Saves state BEFORE search, extracts policy BEFORE restore.

        :param infoset: Current infoset to search from
        :param n_iterations: Number of search iterations (default: self.search_iterations)
        :return: Refined policy for the current infoset
        """
        import copy

        if n_iterations is None:
            n_iterations = getattr(self, 'search_iterations', 100)

        if not getattr(self, 'search_enabled', False):
            return self.get_average_policy(infoset)

        # ======== SAVE STATE BEFORE SEARCH (CRITICAL) ========
        saved_regrets = copy.deepcopy(self.cumulative_regrets)
        saved_policy = copy.deepcopy(self.cumulative_policy)
        saved_iteration_count = self.iteration_count
        # =====================================================

        # Compute posterior ONCE (not every iteration)
        posterior = self.compute_posterior(infoset)

        # Determine which player we are at this infoset
        acting_player = len(infoset.branch) % 2  # 0 = P1, 1 = P2

        # Run online CFR iterations
        for search_iter in range(n_iterations):
            # Sample opponent card from posterior
            opp_card = np.random.choice(len(posterior), p=posterior)

            # Construct state with correct card assignment
            if acting_player == 0:
                state = State(card_p1=infoset.card, card_p2=opp_card, branch=infoset.branch)
            else:
                state = State(card_p1=opp_card, card_p2=infoset.card, branch=infoset.branch)

            # Sample trajectory from this state
            trajectory = Trajectory(state)

            while not self.game.is_terminal(trajectory.state.branch):
                curr_infoset = trajectory.get_current_player_infoset()
                probs = self.get_policy(curr_infoset)  # Current policy (not average)
                actions = list(probs.keys())
                action = random.choices(actions, weights=list(probs.values()))[0]
                trajectory.perform_action(action, action_proba=probs[action])

            # Update regrets (don't increment iteration_count during search)
            # This modifies self.cumulative_regrets and self.cumulative_policy temporarily
            self.update_regrets(trajectory)

        # ======== EXTRACT REFINED POLICY BEFORE RESTORE (CRITICAL) ========
        refined_policy = self.get_average_policy(infoset).copy()
        # ==================================================================

        # ======== RESTORE ORIGINAL STATE (don't pollute training data) ========
        self.cumulative_regrets = saved_regrets
        self.cumulative_policy = saved_policy
        self.iteration_count = saved_iteration_count
        # ======================================================================

        # Re-apply current policy from refined regrets for consistency
        self.set_policy(infoset, self.regret_matching(infoset))

        return refined_policy

    def get_action(self, infoset: Infoset, use_search: bool = None):
        """
        Sample action according to AVERAGE policy, with optional search.

        Use this for EVALUATION only (playing against opponents, measuring exploitability).
        The average policy converges to Nash equilibrium.

        :param use_search: If True, perform online CFR search.
                           If None, uses self.search_enabled default.
        """
        if use_search is None:
            use_search = getattr(self, 'search_enabled', False)

        if use_search and hasattr(self, 'online_cfr_search'):
            refined_policy = self.online_search(infoset)
            actions = list(refined_policy.keys())
            p = list(refined_policy.values())
            return np.random.choice(actions, p=p)
        else:
            probs = self.get_average_policy(infoset)
            actions = list(probs.keys())
            p = list(probs.values())
            return np.random.choice(actions, p=p)

    def set_search_enabled(self, enabled: bool, iterations: int = None):
        """
        Toggle online CFR search and optionally set iteration count.

        :param enabled: Whether to use search during inference
        :param iterations: Number of search iterations (optional)
        """
        self.search_enabled = enabled
        if iterations is not None:
            self.search_iterations = iterations
