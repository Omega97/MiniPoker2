import random
import math
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from time import time
from collections import defaultdict
from contextlib import contextmanager
from mini_poker.game import MiniPoker, Infoset, to_infoset, State, Trajectory, Action, Policy, Logits, Rewards, Visits
from mini_poker.utils import clip
from mini_poker.paths import DATA_DIR
from mini_poker.training.evaluation import evaluate_agents
from mini_poker.utils import print_colored_status


class BaseAgent:
    """
    Generic tabular policy agent.
    Responsibilities:
    - store logits
    - convert logits -> softmax policy
    - sample actions
    """

    def __init__(self,
                 game: MiniPoker,
                 logit_bound=10.,
                 explore_proba: float = None,
                 epochs=None,
                 memory_period=100):
        """
        :param game: game instance
        :param logit_bound: Maximum logit magnitude (for stability)
        :param explore_proba: probability of sampling random action during a rollout
        :param epochs:
        :param memory_period: Number of updates on the reward table after the first update is almost forgotten.
        """
        self.game = game
        self.explore_proba = explore_proba
        self.epochs = epochs
        self.logit_bound = logit_bound
        self.deck_size = game.deck_size
        self.game_tree = game.tree
        self.memory_period = memory_period

        self.logits: Dict[Infoset, Logits] = dict()
        self.policy: Dict[Infoset, Policy] = dict()
        self.training_mode = False
        self.eval_epochs = None
        self.name = None
        self.compare_agent = None
        self.epoch = None
        self.ev_list = None
        self.times = None

        # Modified: Use defaultdict with Rewards and Visits
        self.average_reward: Dict[Infoset, Rewards] = defaultdict(Rewards)
        self.reward_counts: Dict[Infoset, Visits] = defaultdict(Visits)

        self.trajectories_cache: List[Trajectory] = []

        self._init_policy()
        self._init_name()

    def _init_name(self):
        self.name = f"{type(self).__name__}({self.game.game_power},{self.game.deck_size})"
        if self.explore_proba:
            self.name += f"_p{self.explore_proba * 100:.0f}"

    def __str__(self):
        return self.name

    def get_logit(self, infoset: Infoset, action: Action) -> float:
        return self.logits[infoset][action]

    def set_logit(self, infoset: Infoset, action: Action, logit: float):
        value = clip(logit, -self.logit_bound, self.logit_bound)
        self.logits[infoset][action] = value

    def update_logit(self, infoset: Infoset, action: Action, update_step: float):
        """Add update_step and clip for stability."""
        value = self.get_logit(infoset, action) + update_step
        self.set_logit(infoset, action, value)

    def center_logits(self, infoset: Infoset) -> None:
        """Force sum of logits to 0."""
        logits_dict = self.logits[infoset]
        actions = list(logits_dict.keys())

        # Compute mean of current logits
        mean_logit = sum(logits_dict.values()) / len(actions)

        # Subtract mean from each logit (and re-apply clipping)
        for action in actions:
            centered = logits_dict[action] - mean_logit
            self.set_logit(infoset, action, centered)  # set_logit already clips

    def get_logits(self, infoset: Infoset) -> Logits:
        return self.logits[infoset]

    def set_logits(self, infoset: Infoset, logits: Logits):
        """Set logits one by one (with clipping)."""
        if infoset not in self.logits:
            self.logits[infoset] = Logits()

        # Update existing Logits object
        for action, logit in logits.items():
            self.set_logit(infoset, action, logit)

    def get_policy(self, infoset: Infoset) -> Policy:
        return self.policy[infoset]

    def set_policy(self, infoset: Infoset, proba: Policy):
        """Store policy as a Policy object."""
        if isinstance(proba, Policy):
            self.policy[infoset] = proba
        else:
            self.policy[infoset] = Policy(proba)

    def _init_policy(self):
        """Initialize logits and uniform policy for every infoset."""
        for history, actions in self.game_tree.items():
            for card in range(self.game.deck_size):
                infoset = Infoset(card, history)
                # Initialize as Logits object
                self.logits[infoset] = Logits({a: 0.0 for a in actions})
                # Initialize as Policy object
                self.policy[infoset] = Policy({a: 1.0 / len(actions) for a in actions})

    def get_action(self, infoset: Infoset):
        """Sample action according to current policy."""
        probs = self.get_policy(infoset)
        return probs.sample_action()

    def get_average_rewards(self, infoset: Infoset, action: Action) -> float:
        return self.average_reward[infoset][action]

    def get_visit_counts(self, infoset: Infoset, action: Action) -> int:
        return self.reward_counts[infoset][action]

    def softmax_update(self, infoset: Infoset):
        """Convert logits to policy probabilities using Logits.softmax()."""
        logits_obj = self.get_logits(infoset)
        new_policy = logits_obj.softmax()
        self.set_policy(infoset, new_policy)

    def rollout(self, state: State):
        """
        Perform rollout
        Returns (p1_reward, p2_reward)
        """
        temp_state = state
        while temp_state.branch not in self.game.terminals:
            infoset = temp_state.get_current_player_infoset()
            action = self.get_action(infoset)
            temp_state = temp_state.perform_action(action)
        return self.game.get_reward(temp_state)

    def evaluate_action(self, state: State, action, rollout_samples) -> tuple:
        """Evaluate action by averaging rollout rewards."""
        total_p1 = 0
        total_p2 = 0
        for _ in range(rollout_samples):
            temp_state = state
            temp_state = temp_state.perform_action(action)
            r1, r2 = self.rollout(temp_state)
            total_p1 += r1
            total_p2 += r2
        value_p1 = total_p1 / rollout_samples
        value_p2 = total_p2 / rollout_samples
        return value_p1, value_p2

    def _do_explore(self):
        """Randomly generate boolean for exploration."""
        do_explore = False
        if self.explore_proba:
            if random.random() < self.explore_proba:
                do_explore = True
        return do_explore

    def sample_trajectory_logic(self, state) -> Trajectory:
        """
        Sample trajectory using policy with epsilon-greedy exploration.
        """
        trajectory = Trajectory(state)
        while not self.game.is_terminal(trajectory.state.branch):
            infoset = trajectory.get_current_player_infoset()

            # Action selection
            probs = self.policy[infoset]
            actions = list(probs.keys())

            # Epsilon-greedy exploration
            if self._do_explore():
                action = random.choice(actions)
            else:
                action = probs.sample_action()

            trajectory.perform_action(action, action_proba=probs[action])

        return trajectory

    def sample_trajectory_from_root(self, state: State) -> Trajectory:
        """
        Perform and return random trajectory starting from root.
        If training_mode is on, it also memorizes the rewards.
        To change sampling technique, modify 'sample_trajectory_logic'.
        """
        assert state.branch == "", f'Trajectory must start at root state, not "{state.branch}"'

        # Sample trajectory
        trajectory = self.sample_trajectory_logic(state)

        # Add rewards to average
        if self.training_mode:
            self._memorize_rewards(trajectory)
            self.memorize_trajectory(trajectory)

        return trajectory

    def new_reward_weight(self, infoset, action) -> float:
        """
        This weight starts big, and slowly converges to 1-gamma.
        Effectively transitions from *average* to *exponential decay*.
        """
        n = self.get_visit_counts(infoset, action)
        w_new = (1 + n / self.memory_period) / (1 + n)
        return w_new

    def update_average_reward(self, infoset: Infoset, action: Action, new_value: float):
        """Set new average_reward value and update count."""
        self.average_reward[infoset][action] = new_value
        self.reward_counts[infoset].update_count(action, +1)

    def _memorize_rewards(self, trajectory: Trajectory):
        """Apply trajectory rewards on self.average_reward (with discount)."""
        final_rewards = self.game.get_reward(trajectory.state)

        for infoset, action in trajectory.infoset_action_pairs:
            # Modified: No need to check existence, defaultdict handles it
            current_avg = self.get_average_rewards(infoset, action)

            player_idx = infoset.get_current_player()
            w_new = self.new_reward_weight(infoset, action)
            new_avg = current_avg * (1 - w_new) + final_rewards[player_idx] * w_new
            self.update_average_reward(infoset, action, new_value=new_avg)

    def memorize_trajectory(self, trajectory: Trajectory):
        """Add trajectory to the cache."""
        if self.trajectories_cache is None:
            self.trajectories_cache = []
        self.trajectories_cache.append(trajectory)

    def clear_trajectories_cache(self):
        self.trajectories_cache.clear()

    def show_policy(self, logit_mode=False):
        """
        Print learned policies or raw logits.
        :param logit_mode: If True, displays raw logit values instead of percentages.
        """
        mode_title = "LOGITS" if logit_mode else "ACTION PROBABILITIES"
        out = f"\nCOMPLETE LEARNED {mode_title} for N={self.game.game_power} (Stack={self.game.stack})\n\n"
        out += f"{'Card':<5} | {'History':<8} | {mode_title}\n"
        out += "-" * 70 + '\n'

        # Sort histories by length then alphabetical order
        histories = sorted(self.game.tree.keys(), key=lambda x: (len(x), x))

        for history in histories:
            for card in range(self.game.deck_size):
                infoset = Infoset(card, history)

                if logit_mode:
                    # Retrieve raw logit values
                    data = self.get_logits(infoset)
                    data_str = "     ".join(f"{action} {val:>6.2f}" for action, val in data.items())
                else:
                    # Retrieve softmax policy probabilities
                    probs = self.get_policy(infoset)
                    data_str = "     ".join(f"{action} {prob:<4.0%}" for action, prob in probs.items())
                    data_str = data_str.replace(f" {0:.0%}", " - ")

                display_h = history if history else "(root)"
                out += f"{card:<4} | {display_h:<8} |   {data_str}\n"
            out += '\n'
        return out

    def save(self, filepath: str):
        """
        Saves the agent's logits and policy to a JSON file.
        """
        # Convert dictionary keys (Infoset objects) and values (Logits/Policy) to strings/dicts for JSON serialization
        # Modified: Serialize defaultdicts similarly to regular dicts
        data = {
            "logits": {str(list(k)): dict(v) for k, v in self.logits.items()},
            "policy": {str(list(k)): dict(v) for k, v in self.policy.items()},
            "average_reward": {
                str(list(infoset)): {action: val for action, val in actions.items()}
                for infoset, actions in self.average_reward.items()
            },
            "reward_counts": {
                str(list(infoset)): {action: count for action, count in actions.items()}
                for infoset, actions in self.reward_counts.items()
            },
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
        """
        Loads the agent's logits and policy from a JSON file.
        Continues silently if the file does not exist.
        """
        if filepath:
            path = Path(filepath)
        else:
            path = Path(DATA_DIR / f"{self}.json")

        with open(path, 'r') as f:
            data = json.load(f)

        # Load logits as Logits objects
        self.logits = {to_infoset(k): Logits(v) for k, v in data["logits"].items()}
        # Load policy as Policy objects
        self.policy = {to_infoset(k): Policy(v) for k, v in data["policy"].items()}

        # Modified: Reconstruct defaultdicts with Rewards and Visits
        # Reset to empty defaultdicts first
        self.average_reward = defaultdict(Rewards)
        self.reward_counts = defaultdict(Visits)

        if "average_reward" in data:
            for infoset_str, actions in data["average_reward"].items():
                infoset = to_infoset(infoset_str)
                # Create a Rewards object and populate it
                rewards_obj = Rewards()
                for action, value in actions.items():
                    rewards_obj[action] = float(value)
                self.average_reward[infoset] = rewards_obj

        if "reward_counts" in data:
            for infoset_str, actions in data["reward_counts"].items():
                infoset = to_infoset(infoset_str)
                # Create a Visits object and populate it
                visits_obj = Visits()
                for action, count in actions.items():
                    visits_obj[action] = int(count)
                self.reward_counts[infoset] = visits_obj

        print(f"\nAgent {self} loaded from {path}")

    def entropy(self) -> float:
        """
        Calculates the average Shannon entropy across all
        information sets in the current policy.
        """
        if not self.policy:
            return 0.0

        total_entropy = 0.0
        num_infosets = len(self.policy)

        for infoset, probs in self.policy.items():
            infoset_entropy = 0.0
            for prob in probs.values():
                # Entropy is 0 if probability is 0; use a small epsilon or check > 0
                if prob > 0:
                    infoset_entropy -= prob * math.log2(prob)

            total_entropy += infoset_entropy

        return total_entropy / num_infosets

    def terminal_entropy(self) -> float:
        """
        Calculates the average Shannon entropy across only the
        information sets that lead directly to terminal states.
        """
        if not self.policy:
            return 0.0

        total_entropy = 0.0
        terminal_infoset_count = 0

        for infoset, probs in self.policy.items():
            # Check if all possible actions from this history result in a terminal state
            actions = self.game_tree.get(infoset.branch, [])
            if not actions:
                continue

            is_terminal_decision = all(
                (infoset.branch + a) in self.game.terminals
                for a in actions
            )

            if is_terminal_decision:
                infoset_entropy = 0.0
                for prob in probs.values():
                    if prob > 0:
                        infoset_entropy -= prob * math.log2(prob)

                total_entropy += infoset_entropy
                terminal_infoset_count += 1

        if terminal_infoset_count == 0:
            return 0.0

        return total_entropy / terminal_infoset_count

    def best_card_fold_index(self) -> float:
        """
        Returns a metric where:
        - 0.0: Agent never folds the best card.
        - 1.0: Agent is folding the best card at the same rate as a random agent.
        - > 1.0: Agent is actively 'punishing' the best card (worse than random).
        """
        if not self.policy:
            return 0.0

        best_card = self.deck_size - 1
        total_weighted_score = 0.0
        nodes_counted = 0
        fold_action_keys = {'F', 'f'}

        for infoset, probs in self.policy.items():
            if infoset.card == best_card:
                # Check if 'fold' is even an option at this node
                available_actions = list(probs.keys())
                num_actions = len(available_actions)

                # Find the fold action
                fold_prob = 0.0
                has_fold = False
                for action in available_actions:
                    if action in fold_action_keys:
                        fold_prob = probs[action]
                        has_fold = True
                        break

                if has_fold:
                    # Weight by number of actions so (1/N) * N = 1.0
                    total_weighted_score += (fold_prob * num_actions)
                    nodes_counted += 1

        if nodes_counted == 0:
            return -1.0

        return total_weighted_score / nodes_counted

    def inherit_from(self, other_agent):
        # Deep copy to ensure independence
        self.logits = {k: Logits(v) for k, v in other_agent.logits.items()}
        self.policy = {k: Policy(v) for k, v in other_agent.policy.items()}

    def sanity_check(self, rel_tol=1e-5):
        """
        Counts decision-points (infosets) where all possible actions
        have hit the logit boundary (+/- logit_bound).
        """
        saturated_count = 0
        total_infosets = len(self.logits)

        for infoset, action_logits in self.logits.items():
            # Check if every logit in this infoset is at the boundary
            # We use math.isclose to handle potential floating point precision issues
            is_saturated = all(
                math.isclose(abs(logit), self.logit_bound, rel_tol=rel_tol)
                for logit in action_logits.values()
            )

            if is_saturated:
                saturated_count += 1

        print(f"Sanity Check: {saturated_count}/{total_infosets} infosets are fully saturated.")
        return saturated_count

    def set_compare_agent(self, agent, eval_epochs=1):
        """Set agent to compare with during training."""
        self.compare_agent = agent
        self.eval_epochs = eval_epochs

    def training(self):
        """Blanket training method"""
        print(f"\nTraining {self} ...")
        self.ev_list = []
        self.times = [time()]
        print(self.get_progress_bar())
        for self.epoch in range(self.epochs):
            self.training_epoch()
            if self.compare_agent is not None:
                ev_self, ev_other, n_games = evaluate_agents(self.game, [self, self.compare_agent],
                                                             epochs=self.eval_epochs)
                self.ev_list.append(ev_self)
            self.times.append(time())
            print(self.get_progress_bar())

    def get_progress_bar(self, epsilon=1e-6):
        bar = "  "

        # Epochs
        if self.epoch is not None:
            p = (self.epoch + 1) / self.epochs
            bar += f"{self.epoch + 1:3})  {p:8.1%}"
        else:
            bar += f"{0:3})  {0:8.1%}"

        # EVs
        if self.ev_list is not None and len(self.ev_list):
            n = max(1, len(self.ev_list) // 20)  # upper 5%
            ev = np.mean(self.ev_list[-n:])
            bar += print_colored_status(ev, text=f"  {ev:+6.2f}", red=None)
        else:
            bar += f"    ---   "

        # Time
        max_n_times = 5
        if self.times is not None and len(self.times) > 1:
            n_times = min(max_n_times, len(self.times) - 1)
            time_elapsed = self.times[-1] - self.times[-1 - n_times]
            speed = n_times / (time_elapsed + epsilon)
            epochs_left = self.epochs - self.epoch
            time_left = epochs_left / speed
            if time_left > 60:
                bar += f"   eta: {time_left // 60:3.0f} min"
            else:
                bar += f"   eta: {time_left:3.0f} s    "
        else:
            bar += f"   eta: ???      "

        # Exploration
        # Modified: len(self.reward_counts) works on defaultdict
        bar += f"   expl: {len(self.reward_counts) / len(self.policy):7.2%}"

        # Max Magnitude
        # logit_mag = self.average_logit_magnitude()
        # bar += f"   mag:{logit_mag:5.2f}"

        return bar

    def plot_training_ev(self, label="  ", max_window_size=50):
        if not self.ev_list:
            return

        y = np.array(self.ev_list)
        n = len(y)
        y_smooth = np.zeros(n)

        for i in range(n):
            # 1. Determine dynamic window size
            # (Growing window: small at start, large at end)
            win_size = max(1, int(((i + 1) / n) * max_window_size))

            # 2. Slice the data for the current window
            start_idx = max(0, i - win_size + 1)
            window_data = y[start_idx: i + 1]
            current_len = len(window_data)

            if current_len == 1:
                y_smooth[i] = window_data[0]
                continue

            # 3. Create Linear Kernel Weights: y = 2x
            # x is normalized distance within the window [0, 1]
            # We want the most recent point (index -1) to have the highest weight
            weights = np.linspace(0.1, 1.0, current_len)

            # 4. Apply Weighted Average
            y_smooth[i] = np.average(window_data, weights=weights)

        plt.plot(y_smooth, label=label)

    def analyze_infoset_and_posterior(self, infoset: Infoset, exclude_self_card=False):
        """
        Inputs:
          agent: The trained agent (contains the policy).
          my_card: The card currently in your hand.
          history: The string of actions (e.g., "bc").

        Returns:
          action_probs: Probability of actions at the CURRENT state.
          posterior: Distribution over what the opponent is representing.
        """
        game = self.game
        num_cards = game.deck_size

        # 1. Get the current action distribution for OUR hand
        action_probs = self.get_policy(infoset)

        # 2. Calculate Posterior Distribution over Opponent Cards
        # We look at every card the opponent could have and see how likely
        # they were to play the history given the agent's policy.
        opponent_probs = np.zeros(num_cards)

        for card_opp in range(num_cards):

            # Exclude my card
            if exclude_self_card:
                if card_opp == infoset.card:
                    opponent_probs[card_opp] = 0.0
                    continue

            # Probability that opponent reaches this history with this card
            reach_prob = 1.0
            temp_hist = ""

            # Walk through the history string
            for i, action in enumerate(infoset.branch):
                # Check whose turn it was at this step
                # Player 0 acts on even indices, Player 1 on odd
                acting_player = i % 2

                # If it was the OPPONENT's turn to act in the past
                # (Assuming you are player 0 or 1 based on current history length)
                my_player_index = len(infoset.branch) % 2
                if acting_player != my_player_index:
                    # How likely was the opponent to take this specific action?
                    prev_infoset = Infoset(card_opp, temp_hist)
                    probs = self.get_policy(prev_infoset)
                    reach_prob *= probs.get(action, 0.0)

                temp_hist += action

            opponent_probs[card_opp] = reach_prob

        # Normalize to get the posterior (Bayes' Update)
        total_weight = np.sum(opponent_probs)
        if total_weight > 0:
            posterior = opponent_probs / total_weight
        else:
            # If the history is impossible for this policy, return uniform
            posterior = np.ones(num_cards) / (num_cards - 1)
            posterior[infoset.card] = 0

        return action_probs, posterior

    def bayesian_evaluate_action(self, infoset: Infoset, action: Action, n_samples: int = 100) -> tuple:
        """
        Evaluate an action using Bayesian inference over opponent's possible cards.

        Samples opponent cards from the posterior distribution given the current
        infoset and policy, then evaluates the action via rollouts.

        Args:
            infoset: Current information set (card + history)
            action: Action to evaluate
            n_samples: Number of opponent card samples to draw

        Returns:
            tuple: (expected_p1_reward, expected_p2_reward) averaged over samples
        """
        # Get posterior distribution over opponent's possible cards
        _, posterior = self.analyze_infoset_and_posterior(infoset, exclude_self_card=True)

        # Determine which player is acting at this infoset
        acting_player = len(infoset.branch) % 2  # 0 = P1, 1 = P2

        total_p1 = 0.0
        total_p2 = 0.0

        for _ in range(n_samples):
            # Sample opponent card from posterior distribution
            opp_card = np.random.choice(len(posterior), p=posterior)

            # Construct state with correct card assignment based on who is acting
            if acting_player == 0:
                # Current player is P1: infoset.card is P1's card, opp_card is P2's
                state = State(card_p1=infoset.card, card_p2=opp_card, branch=infoset.branch)
            else:
                # Current player is P2: infoset.card is P2's card, opp_card is P1's
                state = State(card_p1=opp_card, card_p2=infoset.card, branch=infoset.branch)

            # Evaluate action with single rollout per sample
            r1, r2 = self.evaluate_action(state, action, rollout_samples=1)
            total_p1 += r1
            total_p2 += r2

        return total_p1 / n_samples, total_p2 / n_samples

    @contextmanager
    def train_context(self):
        """
        Context manager to safely toggle training mode.

        Usage:
            with agent.train_context():
                agent.sample_trajectory(state)
        """
        original_mode = self.training_mode
        self.training_mode = True
        try:
            yield
        finally:
            self.training_mode = original_mode

    def show_average_reward(self, epsilon=1e-3) -> str:
        # 1. Define the Schema: (Column Key, Width, Alignment)
        # Using a list of tuples to keep order and centralize "magic numbers"
        COLUMNS = [
            ("infoset", 23),
            ("action", 5),
            ("value", 10),
            ("visits", 10),
            ("policy", 10),
            ("marks", 6),
        ]

        # Create a reusable format string: "{:<23}{:<5}{:<9}..."
        row_fmt = "".join([f"{{:<{w}}}" for _, w in COLUMNS])

        # Initialize output with the header info
        header_text = f"{len(self.average_reward)} infosets ({len(self.average_reward) / len(self.policy):.2%})"
        out = ["", header_text]

        for infoset, d in self.average_reward.items():
            out.append("")  # Visual break between infosets

            # Pre-calculate "best" benchmarks for the current infoset
            best_value = max(d.values()) if d else 0
            best_visits = max(self.reward_counts[infoset].values()) if self.reward_counts[infoset] else 0
            best_policy = max(self.policy[infoset].values()) if self.policy[infoset] else 0

            actions = self.game_tree[infoset.branch]
            for i, action in enumerate(actions):
                # 2. Extract and format raw data
                val_raw = d.get(action)
                vis_raw = self.reward_counts[infoset].get(action, 0)
                pol_raw = self.policy[infoset].get(action)

                # 3. Apply cell-level display logic
                val_str = f"{val_raw:+.2f}" if val_raw is not None else "-"
                vis_str = f"({vis_raw})" if vis_raw else " - "

                if pol_raw is not None:
                    pol_str = f"{pol_raw:.1%}" if pol_raw > epsilon else "-"
                else:
                    pol_str = " ---"

                # 4. Generate Marks
                mark = ""
                mark += "^" if val_raw == best_value else " "
                mark += "v" if vis_raw == best_visits else " "
                mark += "*" if pol_raw == best_policy else " "

                # 5. Build the row according to the schema
                row_data = [
                    str(infoset) if i == 0 else "",  # Only show infoset on first action line
                    action,
                    val_str,
                    vis_str,
                    pol_str,
                    mark
                ]

                out.append(row_fmt.format(*row_data))

        return "\n".join(out)

    def average_logit_magnitude(self) -> float:
        """
        Calculates the average absolute logit magnitude across all information sets.

        This metric indicates how "confident" or "polarized" the agent's preferences are:
        - Low values (~0): logits are near zero, policy is close to uniform
        - High values (near logit_bound): agent has strong preferences for certain actions

        Returns:
            float: Average of |logit| across all (infoset, action) pairs
        """
        if not self.logits:
            return 0.0

        tot_magnitude = 0.
        n = 0

        for infoset, action_logits in self.logits.items():
            magnitude = 0.
            for logit in action_logits.values():
                magnitude += logit ** 2
            magnitude = magnitude ** 0.5
            tot_magnitude += magnitude
            n += 1

        if n == 0:
            return 0.
        else:
            return tot_magnitude / n

    def normalize_logits(self, epsilon=1e-8):
        """
        For each infoset:
        1. Center logits (subtract mean so they sum to 0)
        2. If L2 norm exceeds logit_bound, scale down so norm equals logit_bound

        This ensures logits remain stable while preserving relative preferences.
        The operation is: logit' = (logit - mean) * min(1, logit_bound / ||centered||_2)
        """
        for infoset, action_logits in self.logits.items():
            actions = list(action_logits.keys())

            # Get current logits as list
            logits = [action_logits[a] for a in actions]

            # Step 1: Center logits (subtract mean)
            mean_logit = sum(logits) / len(logits)
            centered = [logit - mean_logit for logit in logits]

            # Step 2: Compute L2 norm and scale if necessary
            l2_norm = math.sqrt(sum(logit ** 2 for logit in centered))

            if l2_norm > self.logit_bound and l2_norm > epsilon:  # avoid division by zero
                scale = self.logit_bound / l2_norm
                centered = [logit * scale for logit in centered]

            # Update stored logits (set_logit applies individual clipping as safety net)
            for i, action in enumerate(actions):
                self.set_logit(infoset, action, centered[i])

    def training_epoch(self):
        pass
