# training/training_script.py
import random
import matplotlib.pyplot as plt
from typing import Dict
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.cfr_agent import CFRAgent
from mini_poker.agents.new_cfr_agent import NewCFRAgent


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

    name = 'cfr_300'
    agents[name] = CFRAgent(game, epochs=300)
    agents[name].set_compare_agent(agents['crm_5k'])
    AgentTrainer(agents[name]).run(force_training=False)

    name = 'cfr_1k'
    agents[name] = CFRAgent(game, epochs=1000)
    agents[name].set_compare_agent(agents['crm_5k'])
    AgentTrainer(agents[name]).run(force_training=False)

    # # --- New Agent ---
    # name = 'new'
    # agents[name] = NewCFRAgent(game, epochs=100)
    # agents[name].set_compare_agent(agents['crm_5k'])
    # AgentTrainer(agents[name]).run(force_training=True)

    # === Evaluation ===
    print(agents[name].show_policy())
    # print(agents[name].show_average_reward())
    # for name, agent in agents.items():
    #     agent.plot_training_ev(label=name)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
