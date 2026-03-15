from mini_poker.game import MiniPoker
from mini_poker.training.evaluation import all_v_all_tournament
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.kernel_smoothed import KernelSmoothedAgent


def main(game_power=4, deck_size=52, n_games=20_000):
    # Setup
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = list()
    agents.append(BaseAgent(game))

    agent = KernelSmoothedAgent(game, epochs=3_000, lr=0.1, rollout_samples=5, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = KernelSmoothedAgent(game, epochs=10_000, lr=0.001, rollout_samples=1, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = KernelSmoothedAgent(game, epochs=3_000, lr=0.1, rollout_samples=1, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    for s in (0.5, 5.):
        agent = KernelSmoothedAgent(game, epochs=1_000, lr=0.1, rollout_samples=1, explore_proba=0.1, max_sigma=s)
        trainer = AgentTrainer(agent)
        trainer.run()
        agents.append(agent)

    for epochs in (1_000, 10_000, 20_000):
        agent = CounterfactualAgent(game, epochs=epochs, lr=0.001, rollout_samples=2, explore_proba=0.)
        trainer = AgentTrainer(agent)
        trainer.run()
        agents.append(agent)

    agent = CounterfactualAgent(game, epochs=20_000, lr=0.01, rollout_samples=5, explore_proba=0)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = CounterfactualAgent(game, epochs=10_000, lr=0.01, rollout_samples=5, explore_proba=0.1)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = KernelSmoothedAgent(game, epochs=20_000, lr=0.001, rollout_samples=3, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    # Evaluation
    all_v_all_tournament(game, agents, n_games=n_games)


if __name__ == '__main__':
    main()
