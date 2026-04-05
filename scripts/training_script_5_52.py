import random
import matplotlib.pyplot as plt
from typing import Dict
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.cfr_agent import CFRAgent


def main(game_power=5, deck_size=52):
    # --- Setup ---
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # === Agents ===
    agents: Dict[str, BaseAgent] = dict()

    # --- Base Agent ---
    name = 'base'
    agents[name] = BaseAgent(game)

    # --- CRM Agent ---
    name = 'crm_5k'
    agents[name] = CRMAgent(game, epochs=5000, explore_proba=0.1)
    AgentTrainer(agents[name]).run()

    name = 'crm_20k'
    agents[name] = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    AgentTrainer(agents[name]).run()

    # name = 'crm_new'
    # agents[name] = CRMAgent(game, epochs=2000, explore_proba=0.1, warm_up_games=30_000)
    # agents[name].set_compare_agent(agents['crm_5k'])
    # AgentTrainer(agents[name]).run(force_training=True)

    name = 'cfr_new'
    agents[name] = CFRAgent(game, epochs=3000, memory_period=300)
    agents[name].set_compare_agent(agents['crm_5k'])
    AgentTrainer(agents[name]).run(force_training=True)

    # === Evaluation ===
    print()
    # print(agents[name].show_policy())
    print(agents[name].show_average_reward())

    for name, agent in agents.items():
        agent.plot_training_ev(label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
