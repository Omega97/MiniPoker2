from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.training.evaluation import quick_evaluate_agents
from mini_poker.agents.greedy_agent import GreedyAgent
from mini_poker.agents.crm_agent import CRMAgent


def main(n_games=4_000, n_samples=50, explore_proba=0.01):
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


if __name__ == '__main__':
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        print(f"\np = {p:.0%}")
        main(explore_proba=p)
