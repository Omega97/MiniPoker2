# agents\cached_counterfactual_agent.py
import random
import math
from itertools import permutations
from mini_poker.agents.base_agent import BaseAgent, Infoset, State
from mini_poker.game import MiniPoker, Trajectory


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
                 lr=0.01,
                 explore_proba=0.01,
                 memory_period=50,
                 ):
        """
        :param game:
        :param logit_bound:
        :param epochs:
        :param lr:
        :param explore_proba:
        :param memory_period: for how many updates we retain most of the reward info.
        """
        self.epochs = epochs
        self.lr = lr
        self.visited_infosets = dict()

        super().__init__(game,
                         logit_bound,
                         epochs=epochs,
                         explore_proba=explore_proba,
                         memory_period=memory_period)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def get_exploration_weights(self, infoset) -> dict:
        actions = list(self.policy[infoset].keys())
        c = 1.0 / len(actions)
        return {a: c for a in actions}

    def sample_trajectory_logic(self, state) -> Trajectory:
        """
        Sample trajectory using reward-guided exploration.
        During early training, explore more; later, exploit learned policy.
        """
        trajectory = Trajectory(state)

        while not self.game.is_terminal(trajectory.state.branch):
            infoset = trajectory.get_current_player_infoset()
            actions = list(self.policy[infoset].keys())

            # Decide whether to explore or exploit
            if random.random() < self.explore_proba:
                probs = self.get_exploration_weights(infoset)
            else:
                probs = self.policy[infoset]

            action = random.choices(actions, weights=list(probs.values()))[0]

            # Get the probability of this action under current policy
            action_proba = self.policy[infoset].get(action, 0.0)
            trajectory.perform_action(action, action_proba=action_proba)

        return trajectory

    def get_progress_bar(self, epsilon=1e-6):

        # Base bar + entropy + fold best card proba
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"

        # Cache statistics
        total_visits = sum(sum(actions.values()) for actions in self.reward_counts.values())
        bar += f"   Explored: {len(self.average_reward) / len(self.policy):6.2%}"
        bar += f"   Visits: {total_visits}"

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

    def _get_baseline(self, actions, action_values, infoset) -> float:
        """Calculate the baseline (expected value) for the current policy."""
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * action_values[a] for a in actions)
        # baseline = sum(action_values[a] for a in actions) / len(actions)
        return baseline

    def _update_rule(self, actions, action_values, infoset):
        """Direct Update: Update logits based on advantage."""
        baseline = self._get_baseline(actions, action_values, infoset)

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
                self.sample_trajectory_from_root(state)

        # Perform all updates at once
        self.update_visited_infosets()


class NewUpdateRuleAgent(CachedCounterfactualAgent):
    def _update_rule(self, actions, action_values, infoset, k=2.):
        """Direct Update: Update logits based on advantage."""
        max_value = max(action_values.values())
        cutoff = max_value - k

        for action in actions:
            diff = max(0, action_values[action] - cutoff)
            new_logit = diff / k * self.logit_bound * math.tanh(self.epochs/100+1)
            self.set_logit(infoset, action, new_logit)

        # Recalculate probabilities for this infoset
        self.softmax_update(infoset)

    def training_epoch(self, n_times=50):
        """
        Iterates over every initial state to cancel noise.
        The update of the logits happens at the end of the sweep.
        The update of the rewards happens mid-sweep.
        """
        # Training mode ON
        with self.train_context():

            # Iterate over possible initial states
            for _ in range(n_times):
                for card1, card2 in permutations(range(self.deck_size), 2):
                    state = State(card1, card2, branch="")
                    self.sample_trajectory_from_root(state)

        # Perform all updates at once
        self.update_visited_infosets()