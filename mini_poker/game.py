# game.py
from itertools import permutations
from dataclasses import dataclass, replace, field
from typing import List, Tuple, TypeAlias
import ast


# ===== Game Constants =====
LABELS = {2: 'R', 4: 'D', 8: 'T', 16: 'Q', 32: '5', 64: '6', 128: '7'}
INV_LABELS = {name: n for n, name in LABELS.items()}
ACTIONS = tuple(["F", "C"] + list(INV_LABELS) + ["A"])


# ===== Helper dataclasses =====


# Define your alias
Action: TypeAlias = str


@dataclass(frozen=True)
class Infoset:
    card: int
    branch: str

    def get_values(self) -> tuple:
        return self.card, self.branch

    def __repr__(self):
        return f'Infoset({self.card}, "{self.branch}")'

    def __iter__(self):
        yield from self.get_values()

    def get_current_player(self) -> int:
        """Returns the index of the player whose turn it is."""
        return len(self.branch) % 2


@dataclass(frozen=True)
class State:
    card_p1: int
    card_p2: int
    branch: str

    def get_values(self) -> tuple:
        return self.card_p1, self.card_p2, self.branch

    def __repr__(self):
        return f'State({self.card_p1}, {self.card_p2}, "{self.branch}")'

    def __iter__(self):
        yield from self.get_values()

    def __hash__(self):
        return hash(self.get_values())

    def get_current_player(self) -> int:
        """Returns the index of the player whose turn it is."""
        return len(self.branch) % 2

    def get_cards(self) -> tuple[int, int]:
        return self.card_p1, self.card_p2

    def get_current_player_card(self):
        return self.get_cards()[self.get_current_player()]

    def perform_action(self, action: Action) -> 'State':
        """Return a NEW State instance where the move was performed."""
        return replace(self, branch=self.branch + action)

    def get_current_player_infoset(self) -> Infoset:
        card = self.get_current_player_card()
        return Infoset(card, self.branch)


def to_infoset(key_str: str) -> Infoset:
    """
    Helper to reconstruct Infoset from string key:
    [card, 'history']
    """
    card, history = ast.literal_eval(key_str)
    return Infoset(card, history)


@dataclass
class Trajectory:

    # State that evolves throughout the trajectory, and eventually becomes the terminal state
    state: State

    # Records (Infoset, action) for reward tracking
    infoset_action_pairs: List[Tuple[Infoset, str]] = field(default_factory=list)

    # Records (Infoset, (reach_p0, reach_p1)) snapshots
    infoset_proba_pairs: List[Tuple[Infoset, Tuple[float, float]]] = field(default_factory=list)

    # Reach probability for each player
    reach_proba: List[float] = field(default_factory=lambda: [1.0, 1.0])

    def get_state(self) -> State:
        return self.state

    def get_current_player_infoset(self) -> Infoset:
        return self.state.get_current_player_infoset()

    @property
    def total_reach_prob(self) -> float:
        return self.reach_proba[0] * self.reach_proba[1]

    def perform_action(self, action: Action, action_proba: float):
        """
        Records the current state and updates the running reach probability.
        """
        infoset = self.get_current_player_infoset()
        player = infoset.get_current_player()

        # Capture snapshot before update
        self.infoset_proba_pairs.append((infoset, (self.reach_proba[0], self.reach_proba[1])))
        self.infoset_action_pairs.append((infoset, action))

        # Update reach probability
        self.reach_proba[player] *= action_proba

        # Move the game state forward
        self.state = self.state.perform_action(action)

    def get_infosets_history(self):
        cards = self.state.get_cards()
        for i in range(len(self.state.branch)):
            branch = self.state.branch[:i]
            player = len(branch) % 2
            card = cards[player]
            yield Infoset(card, branch)

    def get_state_history(self):
        cards = self.state.get_cards()
        for i in range(len(self.state.branch)):
            branch = self.state.branch[:i]
            yield State(cards[0], cards[1], branch)


# ===== Game class =====


class MiniPoker:

    def __init__(self, game_power, deck_size):
        self.game_power = game_power
        self.deck_size = deck_size
        self.stack = 2 ** game_power
        self.tree = {}
        self.terminals = {}
        self.num_players = 2
        self._build_tree("", 1, 1, 1, 0)

    def _build_tree(self, history, current_bet, p1_comm, p2_comm, acting_p):
        opp_comm, my_comm = self._get_commitments(acting_p, p1_comm, p2_comm)
        actions = self.get_legal_actions(my_comm, opp_comm, current_bet)
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

    def get_legal_actions(self, my_comm, opp_comm, current_bet):
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
            new_bet = current_bet * self._get_multiplier(action)

        if acting_p == 0:
            new_p1, new_p2 = new_bet, p2_comm
        else:
            new_p1, new_p2 = p1_comm, new_bet

        self._build_tree(history, new_bet, new_p1, new_p2, 1 - acting_p)

    def _get_multiplier(self, label):
        """Retrieves the bet multiplier for a given action label."""
        if label == 'A':
            return self.stack
        return INV_LABELS.get(label, 1)

    def get_reward(self, state: State) -> tuple[float, float]:
        """Compute the reward for both players as a tuple: (p1_reward, p2_reward)."""
        p1_comm, p2_comm, is_fold, acting_p = self.terminals[state.branch]

        if is_fold:
            # The acting player is the one who folded
            if acting_p == 0:
                return -p1_comm, p1_comm
            else:
                return p2_comm, -p2_comm

        # Showdown
        if state.card_p1 > state.card_p2:
            return p2_comm, -p2_comm
        elif state.card_p1 < state.card_p2:
            return -p1_comm, p1_comm
        else:
            return 0, 0

    def iter_uniformly_over_hands(self, n_sweeps=1):
        """
        Iterate over card combinations uniformly
        for a certain number of epochs.
        """
        for _ in range(n_sweeps):
            for c1, c2 in permutations(range(self.deck_size), 2):
                yield c1, c2

    def is_terminal(self, branch: str):
        return branch in self.terminals
