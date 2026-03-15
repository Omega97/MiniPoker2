import random
from time import time
from mini_poker.agents.base_agent import BaseAgent, Infoset
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
                 epochs=10_000,
                 lr=0.001,
                 rollout_samples=3,
                 explore_proba=0.,
                 ):
        self.epochs = epochs
        self.lr = lr
        self.rollout_samples = rollout_samples
        self.explore_proba = explore_proba
        self.t0 = None
        super().__init__(game, logit_bound)

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        digits = str(self.lr).split('.')[-1]
        self.name += f"_lr{digits}"
        self.name += f"_e{self.epochs}"
        self.name += f"_r{self.rollout_samples}"
        self.name += f"_p{self.explore_proba * 100:.0f}"

    def sample_trajectories(self, card1, card2, explore=False) -> dict:
        """Perform random trajectory and returns dict of info."""
        visited = {}
        history = ""
        reach_prob = 1.0
        while history not in self.game.terminals:
            player = len(history) % 2
            card = card1 if player == 0 else card2
            infoset = Infoset(card, history)
            visited[infoset] = reach_prob
            if explore:
                actions = self.game.tree[history]
                action = random.choice(actions)
                reach_prob *= (1.0 / len(actions))
            else:
                probs = self.policy[infoset]
                action = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
                reach_prob *= probs[action]
            history += action
        return visited

    def print_progress(self, epoch, t_long=600):
        time_left = None
        tot_epochs = self.epochs * self.deck_size * (self.deck_size - 1)

        if self.t0 is None:
            self.t0 = time()
            self.epoch_start = 0  # Track progress at the last reset
        else:
            time_elapsed = time() - self.t0

            # Check if 10 minutes have passed
            if time_elapsed > t_long:
                self.t0 = time()
                self.epoch_start = epoch
                time_elapsed = 0.001  # Prevent division by zero immediately after reset

            # Calculate speed based on progress since the last reset
            recent_progress = epoch - self.epoch_start
            if time_elapsed > 0:
                speed = recent_progress / time_elapsed
                remaining_epochs = tot_epochs - epoch
                if speed > 0:
                    time_left = remaining_epochs / speed

        p = epoch / tot_epochs
        out = f"\r{epoch:6})  {p:.2%}  S = {self.entropy():.4f}"
        if time_left is not None:
            out += f"   time left: {time_left / 60:.0f} min"
        print(out, end='')

    def train(self, print_period=1000):
        """
        Simplified training loop:
        1. Sample a trajectory.
        2. Calculate advantage for visited nodes.
        3. Update logits.
        """
        for epoch, (card1, card2) in enumerate(self.game.iter_uniformly(self.epochs)):

            if epoch % print_period == 0:
                self.print_progress(epoch)

            # 1. Sample a single trajectory through the game tree
            explore = (random.random() < self.explore_proba)
            visited_infosets = self.sample_trajectories(card1, card2, explore=explore)

            # 2. Update each visited information set
            for (card, history), reach_prob in visited_infosets.items():

                infoset = Infoset(card, history)
                player = len(history) % 2
                actions = self.game.tree[history]

                # Calculate action values using rollouts
                action_values = {}
                for action in actions:
                    rewards = self.evaluate_action(history, action, card1, card2, self.rollout_samples)
                    action_values[action] = rewards[player]

                # Calculate the baseline (expected value) for the current policy
                probs = self.get_policy(infoset)
                baseline = sum(probs[a] * action_values[a] for a in actions)

                # 3. Direct Update: Update logits based on advantage
                for action in actions:
                    # Advantage (NOT weighted by reach probability)
                    advantage = action_values[action] - baseline
                    update_step = self.lr * advantage
                    self.update_logit(infoset, action, update_step)

                # Recalculate probabilities for this infoset
                self.softmax_update(infoset)
