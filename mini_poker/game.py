""" Mini-Poker AI """
from itertools import permutations


LABELS = {2: 'R', 4: 'D', 8: 'T', 16: 'Q', 32: '5', 64: '6', 128: '7'}
INV_LABELS = {name: n for n, name in LABELS.items()}


class MiniPoker:

    def __init__(self, game_power, deck_size):
        self.game_power = game_power
        self.deck_size = deck_size
        self.stack = 2 ** game_power
        self.tree = {}
        self.terminals = {}
        self._build_tree("", 1, 1, 1, 0)

    def _build_tree(self, history, current_bet, p1_comm, p2_comm, acting_p):
        opp_comm, my_comm = self._get_commitments(acting_p, p1_comm, p2_comm)
        actions = self._get_legal_actions(my_comm, opp_comm, current_bet)
        self.tree[history] = actions

        for action in actions:
            new_history = history + action

            if action == 'F':
                self._handle_fold(new_history, p1_comm, p2_comm, acting_p)
            elif action == 'C':
                self._handle_call(new_history, history, current_bet, p1_comm, p2_comm, acting_p)
            else:
                self._handle_bet(new_history, action, current_bet, p1_comm, p2_comm, acting_p)

    @staticmethod
    def _get_commitments(acting_p, p1_comm, p2_comm):
        """Determines the commitment of the opponent and the current player."""
        if acting_p == 0:
            return p2_comm, p1_comm
        else:
            return p1_comm, p2_comm

    def _get_legal_actions(self, my_comm, opp_comm, current_bet):
        """Calculates the list of valid actions based on game state."""
        actions = []
        if my_comm < opp_comm:
            actions.append('F')
        actions.append('C')
        if my_comm < self.stack and opp_comm < self.stack:
            multiplier = 2
            while True:
                raise_to = current_bet * multiplier
                if raise_to <= opp_comm:
                    multiplier *= 2
                    continue
                if raise_to >= self.stack:
                    actions.append('A')
                    break
                label = LABELS.get(multiplier)
                if label:
                    actions.append(label)
                multiplier *= 2
        return actions

    def _handle_fold(self, history, p1_comm, p2_comm, acting_p):
        """Records a terminal state resulting from a Fold."""
        self.terminals[history] = (p1_comm, p2_comm, True, acting_p)

    def _handle_call(self, history, prev_history, current_bet, p1_comm, p2_comm, acting_p):
        """Handles Call logic: matches opponent's commitment."""
        opp_comm, my_comm = self._get_commitments(acting_p, p1_comm, p2_comm)
        new_my_comm = opp_comm

        if acting_p == 0:
            new_p1, new_p2 = new_my_comm, p2_comm
        else:
            new_p1, new_p2 = p1_comm, new_my_comm

        if len(prev_history) > 0:
            self.terminals[history] = (new_p1, new_p2, False, None)
        else:
            self._build_tree(history, current_bet, new_p1, new_p2, 1)

    def _handle_bet(self, history, action, current_bet, p1_comm, p2_comm, acting_p):
        """Handles Raise/All-in logic: updates commitments and recurses."""
        if action == 'A':
            new_bet = self.stack
        else:
            new_bet = current_bet * self._get_mult(action)

        if acting_p == 0:
            new_p1, new_p2 = new_bet, p2_comm
        else:
            new_p1, new_p2 = p1_comm, new_bet

        self._build_tree(history, new_bet, new_p1, new_p2, 1 - acting_p)

    def _get_mult(self, label):
        """Retrieves the bet multiplier for a given action label."""
        if label == 'A':
            return self.stack
        return INV_LABELS.get(label, 1)

    def get_reward(self, history, p1_card, p2_card) -> tuple:
        """Compute the reward for both players as a tuple: (p1_reward, p2_reward)."""
        p1_comm, p2_comm, is_fold, acting_p = self.terminals[history]

        if is_fold:
            # The acting player is the one who folded
            if acting_p == 0:
                return -p1_comm, p1_comm
            else:
                return p2_comm, -p2_comm

        # Showdown
        if p1_card > p2_card:
            return p2_comm, -p2_comm
        elif p1_card < p2_card:
            return -p1_comm, p1_comm
        else:
            return 0, 0

    def iter_uniformly(self, epochs):
        """
        Iterate over card combinations uniformly
        for a certain number of epochs.
        """
        for _ in range(epochs):
            for c1, c2 in permutations(range(self.deck_size), 2):
                yield c1, c2
