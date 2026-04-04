import random
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
    agents = dict()
    agents['base'] = BaseAgent(game)

    # --- EM ---
    name = 'EM_1'
    agents[name] = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=1, explore_proba=1., kernel_size=2)
    AgentTrainer(agents[name]).run()

    name = 'EM_10'
    agents[name] = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=10, explore_proba=1., kernel_size=2)
    AgentTrainer(agents[name]).run()

    name = 'EM_500'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3, explore_proba=1., kernel_size=2)
    AgentTrainer(agents[name]).run()

    name = 'EM_500_2'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3, explore_proba=1., kernel_size=0)
    AgentTrainer(agents[name]).run()

    # --- CRM ---
    name = 'crm_1k'
    agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1)
    AgentTrainer(agents[name]).run()

    name = 'crm_2k'
    agents[name] = CRMAgent(game, epochs=2000, explore_proba=0.01)
    AgentTrainer(agents[name]).run()

    name = 'crm_5k'
    agents[name] = CRMAgent(game, epochs=5000, explore_proba=0.1)
    AgentTrainer(agents[name]).run()

    name = 'crm_20k'
    agents[name] = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    AgentTrainer(agents[name]).run()

    # --- Evaluation ---
    all_v_all_tournament(game, list(agents.values()), epochs=1)


if __name__ == '__main__':
    main()
