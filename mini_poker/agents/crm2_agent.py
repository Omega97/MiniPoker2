# agents\crm_agent.py
from itertools import permutations
import random
import json
import math
import os
from pathlib import Path
from mini_poker.paths import DATA_DIR
from mini_poker.game import to_infoset, Infoset, State, MiniPoker
from mini_poker.agents.base_agent import BaseAgent


class CRMAgent2(BaseAgent):
    """
    Enhanced Counterfactual Regret Minimization Agent (CFR+ with EV-guided exploration).

    Improvements over CRMAgent:
    1. EV-guided exploration: occasionally sample actions by estimated EV
    2. Optimistic regret initialization: small positive bonus encourages early exploration
    3. UCB-style visit bonus: favors under-explored actions
    4. Adaptive decay: exploration parameters decay over training
    5. CFR+ features: linear weighting, regret flooring
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 n_games_compare=10_000,
                 explore_proba=0.1,  # Standard random exploration
                 explore_ev_proba=0.2,  # EV-guided exploration probability
                 ev_blend_alpha=0.8,  # Weight on regret policy vs EV policy
                 optimistic_bonus=0.1,  # Initial regret bonus for all actions
                 ucb_bonus_scale=0.5,  # Scale for UCB-style exploration bonus
                 ev_rollouts=3,  # Rollouts per action for EV estimation
                 ev_temperature=0.5,  # Softmax temperature for EV policy
                 ):
        self.epochs = epochs
        self.explore_proba = explore_proba
        self.explore_ev_proba = explore_ev_proba
        self.ev_blend_alpha = ev_blend_alpha
        self.optimistic_bonus = optimistic_bonus
        self.ucb_bonus_scale = ucb_bonus_scale
        self.ev_rollouts = ev_rollouts
        self.ev_temperature = ev_temperature

        self.cumulative_regrets = {}
        self.cumulative_policy = {}
        self.action_visit_counts = {}
        self.iteration_count = 0

        super().__init__(game, logit_bound, epochs=epochs, n_games_compare=n_games_compare)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        self.name += f"_e{self.epochs}"
        self.name += f"_p{self.explore_proba * 100:.0f}"
        self.name += f"_ev{self.explore_ev_proba * 100:.0f}"

    def _init_policy(self):
        """Initialize with optimistic regrets and visit tracking."""
        for history, actions in self.game_tree.items():
            for card in range(self.deck_size):
                infoset = Infoset(card, history)
                # Optimistic initialization: small positive regret encourages exploration
                self.cumulative_regrets[infoset] = {a: self.optimistic_bonus for a in actions}
                self.cumulative_policy[infoset] = {a: 0.0 for a in actions}
                self.action_visit_counts[infoset] = {a: 0 for a in actions}
                self.set_logits(infoset, {a: 0.0 for a in actions})
                self.set_policy(infoset, {a: 1.0 / len(actions) for a in actions})

    def regret_matching(self, infoset: Infoset) -> dict:
        """CFR+ regret matching with flooring (R+ = max(0, R))."""
        actions = list(self.cumulative_regrets[infoset].keys())
        positive_regrets = {a: max(0, self.cumulative_regrets[infoset][a]) for a in actions}
        sum_positive = sum(positive_regrets.values())

        if sum_positive > 1e-10:
            return {a: r / sum_positive for a, r in positive_regrets.items()}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def _estimate_action_evs(self, infoset: Infoset) -> dict:
        """Quick EV estimation via uniform opponent sampling."""
        actions = list(self.cumulative_regrets[infoset].keys())
        acting_player = len(infoset.branch) % 2
        evs = {}

        for action in actions:
            total = 0.0
            for _ in range(self.ev_rollouts):
                # Sample opponent card uniformly (fast, no recursion)
                opp_card = random.choice([c for c in range(self.deck_size) if c != infoset.card])

                if acting_player == 0:
                    state = State(card_p1=infoset.card, card_p2=opp_card, branch=infoset.branch)
                else:
                    state = State(card_p1=opp_card, card_p2=infoset.card, branch=infoset.branch)

                r1, r2 = self.evaluate_action(state, action, rollout_samples=1)
                total += r1 if acting_player == 0 else r2

            evs[action] = total / self.ev_rollouts

        return evs

    def _ev_guided_policy(self, infoset: Infoset) -> dict:
        """Softmax policy over estimated action EVs."""
        evs = self._estimate_action_evs(infoset)
        actions = list(evs.keys())

        # Numerical stability + temperature scaling
        max_ev = max(evs.values())
        exp_evs = {a: math.exp((v - max_ev) / self.ev_temperature) for a, v in evs.items()}
        z = sum(exp_evs.values())

        return {a: exp_evs[a] / z for a in actions} if z > 1e-10 else {a: 1.0 / len(actions) for a in actions}

    def get_policy(self, infoset: Infoset) -> dict:
        """
        Blended policy: regret-matching + optional EV-guided exploration.
        """
        actions = list(self.cumulative_regrets[infoset].keys())

        # Base: regret-matching policy
        regret_policy = self.regret_matching(infoset)

        # Optional: blend with EV-guided policy
        if self.explore_ev_proba > 0 and random.random() < self.explore_ev_proba:
            ev_policy = self._ev_guided_policy(infoset)
            # Blend: alpha * regret + (1-alpha) * EV
            return {
                a: self.ev_blend_alpha * regret_policy[a] + (1 - self.ev_blend_alpha) * ev_policy[a]
                for a in actions
            }

        return regret_policy

    def get_average_policy(self, infoset: Infoset) -> dict:
        """Get time-averaged policy (CFR convergence target) with CFR+ linear weighting."""
        actions = list(self.cumulative_policy[infoset].keys())
        total = sum(self.cumulative_policy[infoset].values())

        if total > 1e-10:
            return {a: self.cumulative_policy[infoset][a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_action(self, infoset: Infoset):
        """Use time-averaged policy for decisions (CFR theoretical guarantee)."""
        # Standard exploration
        if random.random() < self.explore_proba:
            actions = list(self.policy[infoset].keys())
            return random.choice(actions)

        # Use averaged policy
        probs = self.get_average_policy(infoset)
        r = random.random()
        cumulative = 0.0
        for action in probs:
            cumulative += probs[action]
            if r < cumulative:
                return action
        return list(probs.keys())[-1]

    def sample_trajectory(self, card1, card2) -> tuple:
        """Sample trajectory tracking both players' reach probabilities."""
        visited = {}
        history = ""
        reach_prob_p0, reach_prob_p1 = 1.0, 1.0

        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = (reach_prob_p0, reach_prob_p1)

            probs = self.get_policy(infoset)
            actions = list(probs.keys())

            # Standard random exploration
            if random.random() < self.explore_proba:
                action = random.choice(actions)
            else:
                action = random.choices(actions, weights=list(probs.values()))[0]

            # Track visit counts for UCB bonus
            self.action_visit_counts[infoset][action] += 1

            # Update reach probabilities
            if player == 0:
                reach_prob_p0 *= probs[action]
            else:
                reach_prob_p1 *= probs[action]

            history += action

        return visited, reach_prob_p0, reach_prob_p1

    def get_counterfactual_value(self, state: State, action: str, player: int) -> float:
        """Calculate counterfactual value via rollout."""
        temp_state = state.copy()
        temp_state.branch += action

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
        """Update cumulative regrets with CFR+ features and UCB exploration bonus."""
        visited, reach_prob_p0, reach_prob_p1 = self.sample_trajectory(card1, card2)

        # Current iteration weight for CFR+ linear averaging
        iteration_weight = max(1, self.iteration_count)

        for infoset, (rp0, rp1) in visited.items():
            history = infoset.branch
            player = len(history) % 2

            # Counterfactual reach: opponent's actual reach probability
            cf_reach_prob = rp1 if player == 0 else rp0

            state = State(card1, card2, branch=history)
            actions = self.game.tree[history]

            # Counterfactual values for each action
            action_values = {a: self.get_counterfactual_value(state, a, player) for a in actions}

            # Expected value of current policy
            probs = self.get_policy(infoset)
            policy_value = sum(probs[a] * action_values[a] for a in actions)

            # Update cumulative regrets with UCB-style exploration bonus
            for action in actions:
                regret = action_values[action] - policy_value
                visits = self.action_visit_counts[infoset][action]

                # UCB bonus: higher for less-visited actions, decays with training
                ucb_bonus = self.ucb_bonus_scale / math.sqrt(visits + 1) if visits < 100 else 0

                # CFR+: accumulate with linear weighting
                self.cumulative_regrets[infoset][action] += cf_reach_prob * (regret + ucb_bonus)

            # Update policy via regret matching
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # CFR+: accumulate policy with linear weighting
            for action in new_policy:
                self.cumulative_policy[infoset][action] += new_policy[action] * iteration_weight

    def training_epoch(self):
        """One epoch with adaptive exploration decay."""
        self.iteration_count += 1

        # Adaptive decay of exploration parameters
        progress = min(1.0, self.iteration_count / self.epochs)

        # Decay EV exploration and UCB bonus over time
        self.explore_ev_proba = 0.2 * (1 - progress)  # 0.2 → 0
        self.ucb_bonus_scale = 0.5 * (1 - progress)  # 0.5 → 0

        # Optional: increase EV rollouts late in training for more accurate estimates
        # self.ev_rollouts = 3 + int(2 * progress)  # 3 → 5

        for card1, card2 in permutations(range(self.deck_size), 2):
            self.update_regrets(card1, card2)

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"
        bar += f"   EVexp = {self.explore_ev_proba:.2f}"
        return bar

    def save(self, filepath: str):
        """Save agent with all CFR+ state."""
        data = {
            "logits": {str(list(k)): v for k, v in self.logits.items()},
            "policy": {str(list(k)): v for k, v in self.policy.items()},
            "cumulative_regrets": {str(list(k)): v for k, v in self.cumulative_regrets.items()},
            "cumulative_policy": {str(list(k)): v for k, v in self.cumulative_policy.items()},
            "action_visit_counts": {str(list(k)): v for k, v in self.action_visit_counts.items()},
            "iteration_count": self.iteration_count,
            "explore_proba": self.explore_proba,
            "explore_ev_proba": self.explore_ev_proba,
            "ev_blend_alpha": self.ev_blend_alpha,
            "optimistic_bonus": self.optimistic_bonus,
            "ucb_bonus_scale": self.ucb_bonus_scale,
            "ev_rollouts": self.ev_rollouts,
            "ev_temperature": self.ev_temperature,
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
        """Load agent with all CFR+ state."""
        path = Path(filepath) if filepath else Path(DATA_DIR / f"{self}.json")

        with open(path, 'r') as f:
            data = json.load(f)

        self.logits = {to_infoset(k): v for k, v in data["logits"].items()}
        self.policy = {to_infoset(k): v for k, v in data["policy"].items()}
        self.cumulative_regrets = {to_infoset(k): v for k, v in data["cumulative_regrets"].items()}
        self.cumulative_policy = {to_infoset(k): v for k, v in data["cumulative_policy"].items()}
        self.action_visit_counts = {to_infoset(k): v for k, v in data["action_visit_counts"].items()}
        self.iteration_count = data.get("iteration_count", 0)
        self.explore_proba = data.get("explore_proba", 0.1)
        self.explore_ev_proba = data.get("explore_ev_proba", 0.2)
        self.ev_blend_alpha = data.get("ev_blend_alpha", 0.8)
        self.optimistic_bonus = data.get("optimistic_bonus", 0.1)
        self.ucb_bonus_scale = data.get("ucb_bonus_scale", 0.5)
        self.ev_rollouts = data.get("ev_rollouts", 3)
        self.ev_temperature = data.get("ev_temperature", 0.5)

        print(f"\nAgent {self} loaded from {path}")
