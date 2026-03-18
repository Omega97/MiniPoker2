import matplotlib.pyplot as plt
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.cem_agent import CEMAgent
from mini_poker.agents.cem2_agent import EnhancedCEMAgent


def main(game_power=4, deck_size=52, epochs=30, n_games=20_000):
    # Setup
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = dict()
    agents['base'] = BaseAgent(game)

    # good but the game space is not explored
    agents['good'] = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(agents['good'])
    trainer.run()

    # CEM
    name = 'cem'
    agents[name] = CEMAgent(game, epochs=epochs, lr=0.05, rollout_samples=1,
                            explore_proba=1., max_sigma=10., n_games_compare=n_games)
    agents[name].set_compare_agent(agents['good'])
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # EnhancedCEMAgent
    name = 'cem2'
    agents[name] = EnhancedCEMAgent(game, epochs=epochs, lr=0.05, rollout_samples=1,
                                    explore_proba=1., max_sigma=10., n_games_compare=n_games)
    agents[name].set_compare_agent(agents['good'])
    trainer = AgentTrainer(agents[name])
    trainer.run()

    # Evaluation
    agents['cem2'].plot_training_ev(label='cem2')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
