import random
import matplotlib.pyplot as plt
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent


def main(game_power=4, deck_size=52, n_games=20_000):
    # Setup
    random.seed(42)
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()

    # Good but the game space is not explored
    agent_good = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(agent_good)
    trainer.run()

    # --- CEM 2 ---
    name = 'cem2'
    agents[name] = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                         explore_proba=1., kernel_size=1.)
    agents[name].set_compare_agent(agent_good)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # --- CRM ---
    name = 'crm'
    agents[name] = CRMAgent(game, epochs=300, explore_proba=0.1)
    agents[name].set_compare_agent(agent_good)
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # Evaluation
    for name, agent in agents.items():
        agent.plot_training_ev(label=name)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
