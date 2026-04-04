import numpy as np
import matplotlib.pyplot as plt
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.greedy_agent import GreedyAgent
from mini_poker.training.evaluation import evaluate_agents


def test_1(game_power=5, deck_size=52, epochs=30):
    game = MiniPoker(game_power, deck_size)

    balanced_agent = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    AgentTrainer(balanced_agent).run()

    x_ = np.linspace(0, 2, 21)

    y_ = []
    for x in x_:
        greedy_agent = GreedyAgent(balanced_agent, n_visits=10, k_greed=7.)

        # print(balanced_agent.show_policy())
        # print(balanced_agent.show_average_reward())
        r1, r2, n_games = evaluate_agents(game, agents=[greedy_agent, balanced_agent], epochs=epochs)
        y_.append(r1)
        print(f"({x:.1f}, {r1:+.3f})")
    plt.plot(x_, y_)

    plt.show()


if __name__ == '__main__':
    test_1()
