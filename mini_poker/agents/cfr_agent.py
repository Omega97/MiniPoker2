# agents/simple_cfr_agent.py
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.game import MiniPoker, State, Action, Trajectory, Infoset
import numpy as np


class CFRAgent(CRMAgent):
    """
    CFR Agent with card-similarity kernel propagation.
    High cards (strong hands) get low-variance kernel (stay sharp).
    Low cards get high-variance kernel (more generalization).
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 explore_proba=0.0,
                 kernel_size: float = 0.1):        # 0.0 = no smearing, 1.0 = very strong smearing
        self.kernel_size = kernel_size
        self.kernel = None
        super().__init__(game, logit_bound, epochs=epochs, explore_proba=explore_proba)

        # Cache for exact subgame values
        self._subgame_cache = {}

        # Build the kernel once
        self._compute_kernel()

    def _compute_kernel(self):
        """Exponential kernel: low variance for high cards, high variance for low cards."""
        self.kernel = np.zeros((self.deck_size, self.deck_size))
        for card1 in range(self.deck_size):
            self.kernel[card1, card1] = 1.0
            for card2 in range(card1):
                if self.kernel_size <= 0.0:
                    continue
                # sigma grows as card strength decreases → more smearing for weak hands
                sigma = (self.deck_size - card1) / self.deck_size * self.kernel_size
                diff = abs(card1 - card2) / (self.deck_size - 1)
                weight = np.exp(-diff / sigma)
                self.kernel[card1, card2] = weight
                self.kernel[card2, card1] = weight

    def _compute_subgame_value(self, state: State) -> tuple[float, float]:
        """Exact, memoized expected rewards."""
        key = (state.card_p1, state.card_p2, state.branch)
        if key in self._subgame_cache:
            return self._subgame_cache[key]

        if self.game.is_terminal(state.branch):
            val = self.game.get_reward(state)
            self._subgame_cache[key] = val
            return val

        infoset = state.get_current_player_infoset()
        policy = self.get_policy(infoset)

        ev_p1 = ev_p2 = 0.0
        for action, proba in policy.items():
            if proba <= 0.0:
                continue
            temp_state = state
            temp_state = temp_state.perform_action(action)
            r1, r2 = self._compute_subgame_value(temp_state)
            ev_p1 += proba * r1
            ev_p2 += proba * r2

        val = (ev_p1, ev_p2)
        self._subgame_cache[key] = val
        return val

    def _init_name(self):
        super()._init_name()
        self.name += f"_k{self.kernel_size*100:.0f}"

    def get_counterfactual_value(self, state: State, action: Action, player: int) -> float:
        """Exact cached version."""
        temp_state = state
        temp_state = temp_state.perform_action(action)
        rewards = self._compute_subgame_value(temp_state)
        return rewards[player]

    # ====================== Kernel Propagation ======================

    def _propagate_regret(self, infoset: Infoset, action: Action, regret_delta: float):
        """Propagate regret to similar cards using the kernel."""
        card = infoset.card
        for other_card in range(self.deck_size):
            weight = self.kernel[card, other_card]
            if weight <= 0.0:
                continue
            other_infoset = Infoset(other_card, infoset.branch)
            if other_infoset in self.cumulative_regrets:
                self.cumulative_regrets[other_infoset][action] += weight * regret_delta

    # ====================== Main Update ======================

    def update_regrets(self, trajectory: Trajectory):
        if not trajectory.infoset_proba_pairs:
            return

        card_p1 = trajectory.state.card_p1
        card_p2 = trajectory.state.card_p2

        for infoset, (rp0, rp1) in trajectory.infoset_proba_pairs:
            player = infoset.get_current_player()
            actions = list(self.cumulative_regrets[infoset].keys())

            cf_reach_prob = rp1 if player == 0 else rp0
            temp_state = State(card_p1, card_p2, branch=infoset.branch)

            action_values = {
                a: self.get_counterfactual_value(temp_state, a, player)
                for a in actions
            }

            current_policy = self.get_policy(infoset)
            node_ev = sum(current_policy.get(a, 0.0) * action_values[a] for a in actions)

            # Update regrets + propagate via kernel
            for action in actions:
                regret = action_values[action] - node_ev
                regret_delta = cf_reach_prob * regret

                # Original update
                self.cumulative_regrets[infoset][action] += regret_delta
                self.cumulative_regrets[infoset][action] = max(0.0, self.cumulative_regrets[infoset][action])

                # Propagate to similar cards
                self._propagate_regret(infoset, action, regret_delta)

            # Update current policy
            new_policy = self.regret_matching(infoset)
            self.set_policy(infoset, new_policy)

            # Update average policy
            weight = max(1, self.iteration_count)
            player_reach = rp0 if player == 0 else rp1

            for action in actions:
                self.cumulative_policy[infoset][action] += (
                    player_reach * new_policy[action] * weight
                )

    def training_epoch(self):
        """Clear cache every epoch."""
        self._subgame_cache.clear()
        super().training_epoch()

    def regret_matching(self, infoset: Infoset) -> dict:
        """Standard safe regret matching."""
        actions = list(self.cumulative_regrets[infoset].keys())
        positive = {a: max(0.0, self.cumulative_regrets[infoset][a]) for a in actions}
        total = sum(positive.values())
        if total > 0:
            return {a: positive[a] / total for a in actions}
        c = 1.0 / len(actions)
        return {a: c for a in actions}
