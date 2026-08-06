import numpy as np
import json
import os
import copy
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict

from mini_poker.paths import DATA_DIR
from mini_poker.game import to_infoset, Trajectory, Action, MiniPoker, State, Logits, Policy, Infoset
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.cfr_agent import Regrets


# =============================================================================
# NewCFRAgent – improved version
# =============================================================================
class NewCFRAgent(BaseAgent):
    """
    Improved CFR+ with DCFR discounting.

    Key improvements over the original CFRAgent:
    1. Exact recursive computation of counterfactual values (no Monte-Carlo rollouts).
       → Zero variance in value estimates → dramatically more stable/faster convergence.
    2. Memoized exact value function per hand (cards fixed) for efficiency.
    3. Same outcome-sampling training loop (keeps scalability) but now with perfect
       continuation values.
    4. Online subgame search also benefits from exact values.
    5. Cleaned-up code, better defaults, and consistent Policy handling.
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound: float = 10.0,
                 epochs: int = 1000,
                 memory_period: int = 100,
                 dcfr_alpha: float = 1.5,
                 dcfr_beta: float = 1.0,
                 search_iterations: int = 1000,
                 search_enabled: bool = False):
        self.epochs = epochs

        self.cumulative_regrets: Dict[Infoset, Regrets] = {}
        self.cumulative_policy: Dict[Infoset, Dict[str, float]] = {}

        self.iteration_count = 0
        self.current_player_update = 0
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.search_iterations = search_iterations
        self.search_enabled = search_enabled

        # BaseAgent handles logits/policy/initialization
        super().__init__(game, logit_bound, explore_proba=0.0,
                         epochs=epochs, memory_period=memory_period)

    def _init_name(self):
        self.name = f"NewCFRAgent({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        if self.search_enabled:
            self.name += f"_s{self.search_iterations}"

    # -------------------------------------------------------------------------
    # Metrics (kept for compatibility with training script)
    # -------------------------------------------------------------------------
    def regret_completeness_index(self) -> float:
        total_actions = 0
        visited_actions = 0
        for infoset, regrets in self.cumulative_regrets.items():
            total_actions += len(regrets)
        for infoset, pol_sums in self.cumulative_policy.items():
            for weight in pol_sums.values():
                total_actions += 1
                if weight > 0:
                    visited_actions += 1
        return visited_actions / total_actions if total_actions > 0 else 0.0

    def average_cumulative_regret(self) -> float:
        total = 0.0
        count = 0
        for regrets in self.cumulative_regrets.values():
            for r in regrets.values():
                total += abs(r)
                count += 1
        return total / count if count > 0 else 0.0

    def get_progress_bar(self, epsilon: float = 1e-6) -> str:
        bar = super().get_progress_bar()
        bar += f"   F = {self.best_card_fold_index():.3f}"
        bar += f"   r = {self.regret_completeness_index():7.2%}"
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   |R| = {self.average_cumulative_regret():6.2f}"
        return bar

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, filepath: str):
        def serialize_dict_map(data_map):
            return {str(list(k)): dict(v) for k, v in data_map.items()}

        data = {
            "logits": {str(list(k)): dict(v) for k, v in self.logits.items()},
            "policy": {str(list(k)): dict(v) for k, v in self.policy.items()},
            "cumulative_regrets": serialize_dict_map(self.cumulative_regrets),
            "cumulative_policy": serialize_dict_map(self.cumulative_policy),
            "average_reward": {
                str(list(infoset)): {a: val for a, val in acts.items()}
                for infoset, acts in self.average_reward.items()
            },
            "reward_counts": {
                str(list(infoset)): {a: cnt for a, cnt in acts.items()}
                for infoset, acts in self.reward_counts.items()
            },
            "iteration_count": self.iteration_count,
            "dcfr_params": {"alpha": self.dcfr_alpha, "beta": self.dcfr_beta},
            "search_params": {"iterations": self.search_iterations, "enabled": self.search_enabled},
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
        if filepath:
            path = Path(filepath)
        else:
            path = Path(DATA_DIR / f"{self}.json")

        with open(path, 'r') as f:
            data = json.load(f)

        self.logits = {to_infoset(k): Logits(v) for k, v in data["logits"].items()}
        self.policy = {to_infoset(k): Policy(v) for k, v in data["policy"].items()}

        self.cumulative_regrets = {to_infoset(k): Regrets(v)
                                   for k, v in data["cumulative_regrets"].items()}
        self.cumulative_policy = {to_infoset(k): v
                                  for k, v in data["cumulative_policy"].items()}

        self.iteration_count = data.get("iteration_count", 0)

        if "dcfr_params" in data:
            self.dcfr_alpha = data["dcfr_params"].get("alpha", 1.5)
            self.dcfr_beta = data["dcfr_params"].get("beta", 1.0)
        if "search_params" in data:
            self.search_iterations = data["search_params"].get("iterations", 1000)
            self.search_enabled = data["search_params"].get("enabled", False)

        if "average_reward" in data:
            self.average_reward = defaultdict(lambda: defaultdict(float))
            for infoset_str, acts in data["average_reward"].items():
                infoset = to_infoset(infoset_str)
                for a, val in acts.items():
                    self.average_reward[infoset][a] = float(val)

        if "reward_counts" in data:
            self.reward_counts = defaultdict(lambda: defaultdict(int))
            for infoset_str, acts in data["reward_counts"].items():
                infoset = to_infoset(infoset_str)
                for a, cnt in acts.items():
                    self.reward_counts[infoset][a] = int(cnt)

        print(f"\nAgent {self} loaded from {path}")

    # -------------------------------------------------------------------------
    # Policy / Regret initialization
    # -------------------------------------------------------------------------
    def _init_policy(self):
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                self.cumulative_regrets[infoset] = Regrets({a: 0.0 for a in actions})
                self.cumulative_policy[infoset] = {a: 0.0 for a in actions}
                self.set_logits(infoset, {a: 0.0 for a in actions})
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def get_average_policy(self, infoset: Infoset) -> Policy:
        if infoset not in self.cumulative_policy:
            actions = self.game_tree.get(infoset.branch, [])
            return Policy({a: 1.0 / len(actions) for a in actions}) if actions else Policy()
        return Policy(self.cumulative_policy[infoset]).normalize()

    def regret_matching(self, infoset: Infoset) -> Policy:
        if infoset not in self.cumulative_regrets:
            actions = self.game_tree.get(infoset.branch, [])
            return Policy({a: 1.0 / len(actions) for a in actions}) if actions else Policy()
        return self.cumulative_regrets[infoset].max_normalize()

    # -------------------------------------------------------------------------
    # Exact value computation (the core improvement)
    # -------------------------------------------------------------------------
    def _compute_state_value(self, state: State, player: int,
                             memo: Optional[dict] = None) -> float:
        """
        Exact expected payoff for `player` from `state` when both players follow
        the *current* policy. Uses per-hand memoization.
        """
        if memo is None:
            memo = {}

        key = (state.card_p1, state.card_p2, state.branch)
        if key in memo:
            return memo[key]

        if self.game.is_terminal(state.branch):
            rewards = self.game.get_reward(state)
            val = rewards[player]
            memo[key] = val
            return val

        infoset = state.get_current_player_infoset()
        policy = self.get_policy(infoset)

        ev = 0.0
        for action, proba in policy.items():
            if proba <= 0.0:
                continue
            next_state = state.perform_action(action)
            ev += proba * self._compute_state_value(next_state, player, memo)

        memo[key] = ev
        return ev

    def get_counterfactual_value(self, state: State, action: Action, player: int) -> float:
        """Exact CF-value of deviating to `action` and then following the current policy."""
        next_state = state.perform_action(action)
        return self._compute_state_value(next_state, player)

    # -------------------------------------------------------------------------
    # Training helpers (unchanged structure, now with exact values)
    # -------------------------------------------------------------------------
    def new_policy_weight(self) -> float:
        return max(0, self.iteration_count) + 1

    def apply_dcfr_discount(self, infoset: Infoset):
        if self.iteration_count > 0:
            regret_discount = (self.iteration_count / (self.iteration_count + 1)) ** self.dcfr_alpha
            policy_discount = (self.iteration_count / (self.iteration_count + 1)) ** self.dcfr_beta

            if infoset in self.cumulative_regrets:
                for a in self.cumulative_regrets[infoset]:
                    self.cumulative_regrets[infoset][a] *= regret_discount
            if infoset in self.cumulative_policy:
                for a in self.cumulative_policy[infoset]:
                    self.cumulative_policy[infoset][a] *= policy_discount

    def sample_trajectory_logic(self, state) -> Trajectory:
        trajectory = Trajectory(state)
        while not self.game.is_terminal(trajectory.state.branch):
            infoset = trajectory.get_current_player_infoset()
            probs = self.get_policy(infoset)
            action = probs.sample_action()
            action_proba = probs.get(action, 0.0)
            trajectory.perform_action(action, action_proba=action_proba)
        return trajectory

    def update_regrets(self, trajectory: Trajectory, player_to_update: Optional[int] = None):
        """Update cumulative regrets using *exact* counterfactual values."""
        card_p1 = trajectory.state.card_p1
        card_p2 = trajectory.state.card_p2

        for infoset, (rp0, rp1) in trajectory.infoset_proba_pairs:
            player = infoset.get_current_player()
            if player_to_update is not None and player != player_to_update:
                continue
            if infoset not in self.cumulative_regrets:
                continue

            actions = list(self.cumulative_regrets[infoset].keys())
            cf_reach_prob = rp1 if player == 0 else rp0

            temp_state = State(card_p1, card_p2, branch=infoset.branch)

            self.apply_dcfr_discount(infoset)

            # Exact values – no sampling!
            action_values = {
                a: self.get_counterfactual_value(temp_state, a, player)
                for a in actions
            }

            current_policy = self.get_policy(infoset)
            node_ev = sum(current_policy.get(a, 0.0) * action_values[a] for a in actions)

            for action in actions:
                regret = action_values[action] - node_ev
                weighted_regret = cf_reach_prob * regret

                self.cumulative_regrets[infoset].update_regret(action, weighted_regret)
                self.cumulative_regrets[infoset][action] = max(0.0, self.cumulative_regrets[infoset][action])

            # Update current strategy
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # Update average policy (the Nash target)
            weight = self.new_policy_weight()
            player_reach = rp0 if player == 0 else rp1
            for action in actions:
                self.cumulative_policy[infoset][action] += (
                    player_reach * new_policy.get(action, 0.0) * weight
                )

    def _exploration(self, card1: int, card2: int) -> Trajectory:
        state = State(card1, card2, branch="")
        with self.train_context():
            trajectory = self.sample_trajectory_from_root(state)
        return trajectory

    def training_epoch(self):
        self.current_player_update = self.epoch % 2

        for card1, card2 in self.game.iter_uniformly_over_hands():
            trajectory = self._exploration(card1, card2)
            self.update_regrets(trajectory, player_to_update=self.current_player_update)
            self.iteration_count += 1

        self.clear_trajectories_cache()

    # -------------------------------------------------------------------------
    # Online subgame solving (also benefits from exact values)
    # -------------------------------------------------------------------------
    def compute_posterior(self, infoset: Infoset) -> np.ndarray:
        """Posterior over opponent cards given the observed history (using average policy)."""
        num_cards = self.game.deck_size
        opponent_probs = np.zeros(num_cards)

        for card_opp in range(num_cards):
            if card_opp == infoset.card:
                opponent_probs[card_opp] = 0.0
                continue

            reach_prob = 1.0
            temp_hist = ""

            for i, action in enumerate(infoset.branch):
                acting_player = i % 2
                my_player_index = len(infoset.branch) % 2
                if acting_player != my_player_index:
                    prev_infoset = Infoset(card_opp, temp_hist)
                    if prev_infoset in self.cumulative_policy:
                        probs = self.get_average_policy(prev_infoset)
                        reach_prob *= probs.get(action, 0.0)
                    else:
                        actions_list = self.game_tree.get(prev_infoset.branch, [])
                        reach_prob *= 1.0 / len(actions_list) if actions_list else 1.0
                temp_hist += action

            opponent_probs[card_opp] = reach_prob

        total = np.sum(opponent_probs)
        if total > 1e-10:
            return opponent_probs / total
        else:
            posterior = np.ones(num_cards) / (num_cards - 1)
            posterior[infoset.card] = 0.0
            return posterior

    def online_search(self, infoset: Infoset, n_iterations: Optional[int] = None) -> Policy:
        """Perform online CFR subgame solving from the current infoset."""
        if n_iterations is None:
            n_iterations = self.search_iterations

        if not self.search_enabled:
            return self.get_average_policy(infoset)

        # Save everything before temporary search
        saved_regrets = copy.deepcopy(self.cumulative_regrets)
        saved_policy = copy.deepcopy(self.cumulative_policy)
        saved_iteration_count = self.iteration_count

        posterior = self.compute_posterior(infoset)
        acting_player = len(infoset.branch) % 2

        for _ in range(n_iterations):
            opp_card = np.random.choice(len(posterior), p=posterior)

            if acting_player == 0:
                state = State(infoset.card, opp_card, infoset.branch)
            else:
                state = State(opp_card, infoset.card, infoset.branch)

            # Sample one trajectory from this infoset onward
            trajectory = Trajectory(state)
            while not self.game.is_terminal(trajectory.state.branch):
                curr_infoset = trajectory.get_current_player_infoset()
                probs = self.get_policy(curr_infoset)
                action = probs.sample_action()
                action_proba = probs.get(action, 0.0)
                trajectory.perform_action(action, action_proba=action_proba)

            self.update_regrets(trajectory)   # exact updates

        # Extract refined policy
        refined_policy = self.get_average_policy(infoset).copy()

        # Restore original training state
        self.cumulative_regrets = saved_regrets
        self.cumulative_policy = saved_policy
        self.iteration_count = saved_iteration_count

        self.set_policy(infoset, self.regret_matching(infoset))

        return refined_policy

    def get_action(self, infoset: Infoset, use_search: Optional[bool] = None) -> Action:
        """Action selection for evaluation / play."""
        if use_search is None:
            use_search = self.search_enabled

        if use_search:
            refined = self.online_search(infoset)
            return refined.sample_action()
        else:
            probs = self.get_average_policy(infoset)
            return probs.sample_action()

    def set_search_enabled(self, enabled: bool, iterations: Optional[int] = None):
        self.search_enabled = enabled
        if iterations is not None:
            self.search_iterations = iterations
