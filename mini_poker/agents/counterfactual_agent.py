import random
from typing import Dict
from mini_poker.agents.base_agent import BaseAgent, Infoset, State
from mini_poker.game import MiniPoker


class CounterfactualAgent(BaseAgent):
    """
    Counterfactual expectation maximization
    A streamlined version of the CounterfactualAgent.
    Removes momentum and batching for a direct 'REINFORCE-style'
    update to the tabular policy.
    """
    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 epochs=1_000,
                 lr=0.01,
                 rollout_samples=1,
                 explore_proba=0.01,
                 ):
        self.epochs = epochs
        self.lr = lr
        self.rollout_samples = rollout_samples
        self.explore_proba = explore_proba
        self.visited_infosets = None
        super().__init__(game, logit_bound, epochs=epochs)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_r{self.rollout_samples}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def sample_trajectory_from_root(self, state: State) -> dict:
        """Perform random trajectory and returns dict of info."""
        card1, card2 = state.get_cards()
        history = ""
        visited = {}
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            probs = self.policy[infoset]
            action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
            reach_prob *= probs[action]
            history += action
        return visited

    def sample_random_trajectory(self, state: State) -> Dict[Infoset, float]:
        """Perform random trajectory and return dict of info."""
        card1, card2 = state.get_cards()
        history = ""
        visited = {}
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            actions = self.game.tree[history]
            action = random.choice(actions)
            reach_prob *= (1.0 / len(actions))
            history += action
        return visited

    def get_progress_bar(self, epsilon=1e-6):
        bar = super().get_progress_bar()
        bar += f"   S = {self.entropy():.4f}"
        bar += f"   St = {self.terminal_entropy():.4f}"
        bar += f"   F = {self.best_card_fold_index():.4f}"

    def explore_trajectory(self, state):
        """ Sample a single trajectory through the game tree. """
        explore = (random.random() < self.explore_proba)
        if explore:
            self.visited_infosets = self.sample_random_trajectory(state)
        else:
            self.visited_infosets = self.sample_trajectory_from_root(state)

    def get_baseline(self, actions, action_values, infoset) -> float:
        # Calculate the baseline (expected value) for the current policy
        probs = self.get_policy(infoset)
        baseline = sum(probs[a] * action_values[a] for a in actions)
        return baseline

    def _update_rule(self, actions, action_values, infoset):
        """ Direct Update: Update logits based on advantage. """
        baseline = self.get_baseline(actions, action_values, infoset)

        for action in actions:
            # Advantage (NOT weighted by reach probability)
            advantage = action_values[action] - baseline
            update_step = self.lr * advantage
            self.update_logit(infoset, action, update_step)

        # Recalculate probabilities for this infoset
        self.softmax_update(infoset)

    def get_action_values(self, actions, history, card1, card2) -> dict:
        """ Calculate action values using rollouts. """
        player = len(history) % 2
        action_values = {}
        for action in actions:
            state = State(card1, card2, branch=history)
            rewards = self.evaluate_action(state, action, self.rollout_samples)
            action_values[action] = rewards[player]
        return action_values

    def update_visited_infosets(self, state):
        """ Update each visited information set. """
        card1, card2 = state.get_cards()
        for (card, history), reach_prob in self.visited_infosets.items():
            infoset = Infoset(card, history)
            actions = self.game.tree[history]
            action_values = self.get_action_values(actions, history, card1, card2)
            self._update_rule(actions, action_values, infoset)

    def training_epoch(self):
        """
        Simplified training loop:
        1. Sample a trajectory.
        2. Calculate advantage for visited nodes.
        3. Update logits.
        """
        for iteration, (card1, card2) in enumerate(self.game.iter_uniformly_over_hands(self.epochs)):
            state = State(card1, card2, branch="")
            self.explore_trajectory(state)
            self.update_visited_infosets(state)
