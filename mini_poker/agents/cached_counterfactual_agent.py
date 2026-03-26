# agents\cached_counterfactual_agent.py
import random
from itertools import permutations
from mini_poker.agents.base_agent import BaseAgent, Infoset, State
from mini_poker.game import MiniPoker


class CachedCounterfactualAgent(BaseAgent):
    """
    Counterfactual expectation maximization with cached rewards.

    Similar to CounterfactualAgent but uses self.average_reward
    instead of rollouts for action value estimation. Much faster!

    Key features:
    1. Two-phase training: exploration → exploitation
    2. Uses cached average rewards instead of Monte Carlo rollouts
    3. Falls back to random values if cache is empty
    4. Progressive cache warming with decay factor
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 n_games_compare=10_000,
                 lr=0.01,
                 explore_proba=0.01,
                 memory_period=50,
                 ):
        """

        :param game:
        :param logit_bound:
        :param epochs:
        :param n_games_compare:
        :param lr:
        :param explore_proba:
        :param memory_period: for how many updates we retain most of the reward info.
        """
        self.epochs = epochs
        self.lr = lr
        self.explore_proba = explore_proba
        self.visited_infosets = dict()

        super().__init__(game,
                         logit_bound,
                         epochs=epochs,
                         n_games_compare=n_games_compare,
                         memory_period=memory_period)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def sample_trajectory(self, state: State):
        """Perform random trajectory with epsilon-greedy exploration.

        With probability explore_proba: take a random action (uniform)
        With probability 1-explore_proba: take an action from current policy
        """
        card1, card2 = state.get_cards()
        history = ""
        reach_prob = 1.0

        path = []

        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            self.visited_infosets[infoset] = reach_prob
            probs = self.policy[infoset]
            actions = list(probs.keys())

            # Epsilon-greedy action selection
            if random.random() < self.explore_proba:
                action = random.choice(actions)
            else:
                action = random.choices(actions, weights=list(probs.values()))[0]

            reach_prob *= probs[action]
            history += action
            path.append((infoset, action))

        if self.training_mode:
            final_state = State(card1, card2, history)
            self._memorize_rewards(path, final_state)


    def get_progress_bar(self, epsilon=1e-6):

        # Base bar + entropy + fold best card proba
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"

        # Cache statistics
        total_visits = sum(sum(actions.values()) for actions in self.reward_counts.values())
        bar += f"   Cache = {len(self.average_reward)}"
        bar += f"   Visits = {total_visits}"

        return bar

    def get_cached_action_values(self, actions, infoset) -> dict:
        """
        Get action values from cache instead of rollouts.
        Falls back to random estimates if cache is insufficiently populated.
        """
        action_values = {}

        for action in actions:
            action_values[action] = self.average_reward[infoset][action]

        return action_values

    def get_baseline(self, actions, action_values, infoset) -> float:
        """Calculate the baseline (expected value) for the current policy."""
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * action_values[a] for a in actions)
        return baseline

    def _update_rule(self, actions, action_values, infoset):
        """Direct Update: Update logits based on advantage."""
        baseline = self.get_baseline(actions, action_values, infoset)

        for action in actions:
            # Advantage (NOT weighted by reach probability)
            advantage = action_values[action] - baseline
            update_step = self.lr * advantage
            self.update_logit(infoset, action, update_step)

        # Recalculate probabilities for this infoset
        self.softmax_update(infoset)

    def update_visited_infosets(self):
        """Update each visited information set using cached or rollout values."""
        for (card, history), reach_prob in self.visited_infosets.items():
            infoset = Infoset(card, history)
            actions = self.game.tree[history]
            action_values = self.get_cached_action_values(actions, infoset)
            self._update_rule(actions, action_values, infoset)

        # Clear visits cache
        self.visited_infosets = {}

    def training_epoch(self):
        """
        Iterates over every initial state to cancel noise.
        The update of the logits happens at the end of the sweep.
        The update of the rewards happens mid-sweep.
        """
        # Training mode ON
        with self.train_context():

            # Iterate over possible initial states
            for card1, card2 in permutations(range(self.deck_size), 2):
                state = State(card1, card2, branch="")
                self.sample_trajectory(state)

        # Perform all updates at once
        self.update_visited_infosets()
