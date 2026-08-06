import random
from typing import Dict
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.training.evaluation import all_v_all_tournament
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.cfr_agent import CFRAgent


def main(game_power=5, deck_size=52):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents: Dict[str: BaseAgent] = dict()
    agents['base'] = BaseAgent(game)

    # --- CRM ---
    # name = 'crm_1k'
    # agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1)
    # AgentTrainer(agents[name]).run()

    # name = 'crm_2k'
    # agents[name] = CRMAgent(game, epochs=2000, explore_proba=0.01)
    # AgentTrainer(agents[name]).run()

    name = 'crm_5k'
    agents[name] = CRMAgent(game, epochs=5000, explore_proba=0.1)
    AgentTrainer(agents[name]).run()

    name = 'crm_20k'
    agents[name] = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    AgentTrainer(agents[name]).run()

    # --- CFR ---
    # name = 'cfr_300'
    # agents[name] = CFRAgent(game, epochs=300)
    # AgentTrainer(agents[name]).run()

    name = 'cfr_1k'
    agents[name] = CFRAgent(game, epochs=1000)
    AgentTrainer(agents[name]).run()

    name = 'cfr_1k_search'
    agents[name] = CFRAgent(game, epochs=1000, search_enabled=True, search_iterations=10, n_moves_no_search=2)
    agents[name].inherit_from(agents['cfr_1k'])

    # name = 'cfr_1500'
    # agents[name] = CFRAgent(game, epochs=1500)
    # AgentTrainer(agents[name]).run()
    #
    # name = 'cfr_2k'
    # agents[name] = CFRAgent(game, epochs=2000)
    # AgentTrainer(agents[name]).run()
    #
    # name = 'cfr_3k'
    # agents[name] = CFRAgent(game, epochs=3000)
    # AgentTrainer(agents[name]).run()

    # --- Evaluation ---
    all_v_all_tournament(game, list(agents.values()), epochs=1)


if __name__ == '__main__':
    main()
