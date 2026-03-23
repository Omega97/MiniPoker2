import random
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.training.evaluation import all_v_all_tournament


def main(game_power=5, deck_size=52, n_games=20_000):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()
    agents['base'] = BaseAgent(game)

    # --- CRM ---
    name = 'crm_1000'
    agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

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

    # --- Evaluation ---
    all_v_all_tournament(game, list(agents.values()), n_games=50_000)


if __name__ == '__main__':
    main()
