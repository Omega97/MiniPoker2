import random
import matplotlib.pyplot as plt
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent


def main(game_power=5, deck_size=52, n_games=20_000):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()

    # --- CRM ---
    name = 'crm_1000'
    agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1, n_games_compare=n_games)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # --- EM ---
    name = 'EM'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.005, rollout_samples=10,
                                         explore_proba=1., max_sigma=2., n_games_compare=n_games)
    trainer = AgentTrainer(agents[name], force_training=True)
    agents[name].set_compare_agent(agents['crm_1000'])
    trainer.run()

    # --- Evaluation ---
    for name, agent in agents.items():
        agent.plot_training_ev(label=name)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
