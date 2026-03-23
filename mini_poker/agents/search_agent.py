# agents\terminal_optimal_em_agent.py
import random
from typing import Optional
from mini_poker.game import Infoset, State
from mini_poker.agents.cem2_agent import CounterfactualEMAgent


class TerminalOptimalEMAgent(CounterfactualEMAgent):
    """
    Counterfactual EM Agent with optimal terminal node decisions.

    At terminal decision points (where only Fold/Call remain), calculates
    exact EV of each action and picks the best one deterministically.

    This is safe because:
    1. No opponent can respond after this decision
    2. We're not leaking information about our strategy
    3. Pure exploitation at terminal nodes doesn't hurt equilibrium
    """

    def _init_name(self):
        super()._init_name()
        self.name += "_term"

    def is_terminal_decision(self, infoset: Infoset) -> bool:
        """
        Check if all actions from this infoset lead to terminal states.

        Terminal decisions are those where the opponent cannot respond
        (e.g., facing an all-in, or closing the betting round).
        """
        actions = self.game.tree[infoset.branch]
        if not actions:
            return False

        return all(
            (infoset.branch + action) in self.game.terminals
            for action in actions
        )

    def calculate_terminal_ev(self, state: State, action: str) -> float:
        """
        Calculate exact EV for terminal actions (Fold/Call).

        Args:
            state: Full game state (both cards known)
            action: The action to evaluate ('F' or 'C')

        Returns:
            Expected value for the current player
        """
        player = len(state.branch) % 2
        my_card = state.card_p1 if player == 0 else state.card_p2

        if action == 'F':
            # Fold: lose what you've committed
            p1_comm, p2_comm, is_fold, acting_p = self.game.terminals[state.branch + action]
            if player == 0:
                return -p1_comm
            else:
                return -p2_comm

        elif action == 'C':
            # Call: calculate EV based on posterior of opponent's hand
            opponent_posterior = self._get_posterior(my_card, state.branch)

            # Get the pot size (total commitments after call)
            p1_comm, p2_comm, is_fold, acting_p = self.game.terminals[state.branch + action]

            ev = 0.0
            for opp_card in range(self.deck_size):
                if opp_card == my_card:
                    continue

                # Probability opponent has this card
                opp_prob = opponent_posterior[opp_card]

                # Determine winner and payoff
                if player == 0:
                    # I'm P1
                    if my_card > opp_card:
                        ev += opp_prob * p2_comm  # Win opponent's commitment
                    elif my_card < opp_card:
                        ev += opp_prob * (-p1_comm)  # Lose my commitment
                    else:
                        ev += opp_prob * 0  # Tie
                else:
                    # I'm P2
                    if my_card > opp_card:
                        ev += opp_prob * p1_comm  # Win opponent's commitment
                    elif my_card < opp_card:
                        ev += opp_prob * (-p2_comm)  # Lose my commitment
                    else:
                        ev += opp_prob * 0  # Tie

            return ev

        return 0.0

    def get_action(self, infoset: Infoset, state: Optional[State] = None):
        """
        Override to use optimal decisions at terminal nodes.

        At terminal nodes: Calculate exact EV and pick best action
        At non-terminal nodes: Sample from learned policy

        Args:
            infoset: Current information set (card + history)
            state: Optional full game state (needed for terminal EV calculation)
        """
        # Check if this is a terminal decision point
        if self.is_terminal_decision(infoset) and state is not None:
            actions = self.game.tree[infoset.branch]

            # Only apply to Fold/Call decisions
            if all(a in ['F', 'C'] for a in actions):
                # Calculate exact EV for each action
                action_evs = {}
                for action in actions:
                    action_evs[action] = self.calculate_terminal_ev(state, action)

                # Pick the action with highest EV
                best_action = max(action_evs, key=action_evs.get)
                return best_action

        # Default: sample from policy
        probs = self.get_policy(infoset)
        r = random.random()
        cumulative = 0.0
        for action in probs:
            cumulative += probs[action]
            if r < cumulative:
                return action
        return list(probs.keys())[-1]

    def rollout(self, state: State):
        """
        Override rollout to pass state to get_action for terminal EV calculation.
        """
        while state.branch not in self.game.terminals:
            player = len(state.branch) % 2
            card = state.card_p1 if player == 0 else state.card_p2
            infoset = Infoset(card, state.branch)
            # Pass state so terminal nodes can calculate exact EV
            action = self.get_action(infoset, state=state)
            state.branch += action
        return self.game.get_reward(state)
