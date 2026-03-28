import random
import matplotlib.pyplot as plt
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.cached_counterfactual_agent import CachedCounterfactualAgent
from mini_poker.agents.crm_agent import CRMAgent


def main(game_power=5, deck_size=52):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()

    # --- CRM ---
    name = 'crm_1000'
    agents[name] = CRMAgent(game, epochs=1000, explore_proba=0.1)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # name = 'crm_c'
    # agents[name] = CRMAgent(game, epochs=5000, explore_proba=0.1, n_games_compare=n_games)
    # trainer = AgentTrainer(agents[name])
    # trainer.run()

    # --- EM ---
    name = 'EM'
    agents[name] = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3, explore_proba=1., max_sigma=.0)
    AgentTrainer(agents[name]).run()

    # --- CC ---
    name = 'cc'
    agents[name] = CachedCounterfactualAgent(game, epochs=2_000, lr=0.01, explore_proba=0.1, n_games_compare=1_000)
    agents[name].set_compare_agent(agents['crm_1000'])
    AgentTrainer(agents[name]).run()

    # new
    name = 'new'
    agents[name] = CRMAgent(game, epochs=500, explore_proba=0.1, n_games_compare=2_000)
    agents[name].set_compare_agent(agents['crm_1000'])
    AgentTrainer(agents[name], force_training=True).run()

    print()
    print(agents[name].show_average_reward())

    # --- Evaluation ---
    for name, agent in agents.items():
        agent.plot_training_ev(label=name)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
