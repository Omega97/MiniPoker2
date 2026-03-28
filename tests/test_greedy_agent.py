import random
import numpy as np
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.training.evaluation import quick_evaluate_agents
from mini_poker.agents.greedy_agent import GreedyAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.cached_counterfactual_agent import CachedCounterfactualAgent


class HybridExploitAgent(CachedCounterfactualAgent):
    """Uses trained policy but occasionally picks max-EV action."""

    def __init__(self, base_agent, p_exploit=0.10):
        # Copy all attributes from trained agent
        self.__dict__.update(base_agent.__dict__)
        self.p_exploit = p_exploit

    def get_action(self, infoset):
        """With probability p_exploit, pick max-EV action; otherwise follow policy."""
        if random.random() < self.p_exploit:
            # Greedy: pick action with highest cached EV
            return self._get_max_ev_action(infoset)
        else:
            # Follow trained policy
            return super().get_action(infoset)

    def _get_max_ev_action(self, infoset):
        """Return action with highest cached average reward."""
        actions = list(self.policy[infoset].keys())
        best_action = None
        best_value = float('-inf')

        for action in actions:
            # Use cached average reward as EV estimate
            visits = self.reward_counts[infoset][action]
            if visits > 0:
                value = self.average_reward[infoset][action]
            else:
                # Optimistic bonus for unvisited actions
                value = 1.0

            if value > best_value:
                best_value = value
                best_action = action

        return best_action if best_action else random.choice(actions)


def test_1(n_games=2_000, n_samples=50, explore_proba=0.01):
    game = MiniPoker(5, 52)

    # --- CRM ---
    crm_agent = CRMAgent(game, epochs=5000, explore_proba=0.1)
    trainer = AgentTrainer(crm_agent)
    trainer.run()

    # --- Greedy ---
    greedy = GreedyAgent(game, n_samples=n_samples, explore_proba=explore_proba)
    greedy.inherit_from(crm_agent)

    # Match
    r1, r2 = quick_evaluate_agents(game, greedy, crm_agent, n_games)
    print(f"reward = {r1:.2f}")


def test_2(n_games=100, p_exploit=.3):
    game = MiniPoker(5, 52)

    # --- CRM ---
    crm_agent = CRMAgent(game, epochs=5000, explore_proba=0.1)
    AgentTrainer(crm_agent).run()

    # --- Greedy ---
    greedy = GreedyAgent(game, n_samples=50, explore_proba=0.33)
    greedy.inherit_from(crm_agent)

    # --- Cached Counterfactual ---
    cached_agent = CachedCounterfactualAgent(game, epochs=2_000, lr=0.01, explore_proba=0.1)
    AgentTrainer(cached_agent).run()

    # --- Create Hybrid Agent: Cached policy + p_exploit greedy action selection ---

    # Create hybrid agent from trained cached agent
    hybrid_agent = HybridExploitAgent(cached_agent, p_exploit=p_exploit)

    # --- Match: Hybrid vs CRM ---
    print(f"Hybrid (p_exploit={p_exploit:.0%})\n")
    i = 0
    games_ = []
    rewards_ = []
    while True:
        i += 1
        new_games = n_games * i
        r_hybrid, r_2 = quick_evaluate_agents(game, greedy, crm_agent, new_games)
        games_.append(new_games)
        rewards_.append(r_hybrid)
        estimate = np.array(games_).dot(np.array(rewards_)) / sum(games_)
        print(f"{sum(games_):8})  reward = {estimate:+.3f}")


if __name__ == '__main__':
    # test_1()
    test_2()
