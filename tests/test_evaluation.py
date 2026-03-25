import random
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.training.evaluation import all_v_all_tournament
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.decisive_agent import DecisiveAgent
from mini_poker.agents.greedy_agent import GreedyAgent


def main(game_power=5, deck_size=52, n_games=20_000):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()
    agents['base'] = BaseAgent(game)

    # --- EM ---
    name = 'EM_1'
    agents[name] = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=1,
                                         explore_proba=1., max_sigma=2., n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'EM_10'
    agents[name] = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=10,
                                         explore_proba=1., max_sigma=2., n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'EM_500'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3,
                                         explore_proba=1., max_sigma=2., n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'EM_500_2'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3,
                                         explore_proba=1., max_sigma=0., n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # --- CRM ---
    name = 'crm_1'
    agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'crm_2'
    agents[name] = CRMAgent(game, epochs=2000, explore_proba=0.1, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'crm_3'
    agents[name] = CRMAgent(game, epochs=2000, explore_proba=0.01, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    name = 'crm_5'
    agents[name] = CRMAgent(game, epochs=5000, explore_proba=0.1, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # --- Decisive ---
    name = 'decisive_5'
    agents[name] = DecisiveAgent(game, epsilon=1e-3)
    agents[name].inherit_from(agents['crm_5'])

    # --- Evaluation ---
    all_v_all_tournament(game, list(agents.values()), n_games=100)


if __name__ == '__main__':
    main()
