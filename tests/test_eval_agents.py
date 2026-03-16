from mini_poker.game import MiniPoker
from mini_poker.training.evaluation import all_v_all_tournament
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.kernel_smoothed import KernelSmoothedAgent
from mini_poker.agents.batch_kernel_smoothed_agent import BatchKernelSmoothedAgent
from mini_poker.agents.batch_kernel_smoothed_methodical_agent import BatchKernelSmoothedMethodicalAgent
from mini_poker.agents.crm_agent import CounterfactualRegretMinimizationAgent
from mini_poker.agents.new_agent import NewAgent
from mini_poker.paths import DATA_DIR


def main(game_power=4, deck_size=52, n_games=50_000):
    # Setup
    game = MiniPoker(game_power, deck_size)

    # Add agents
    agents = list()
    agents.append(BaseAgent(game))

    agent = KernelSmoothedAgent(game, epochs=3_000, lr=0.1, rollout_samples=5, explore_proba=0.1, max_sigma=5.)
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

    # good but the game space is not explored
    for epochs in (10_000, 20_000):
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

    agent = KernelSmoothedAgent(game, epochs=10_000, lr=0.001, rollout_samples=1, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    # good
    agent = BatchKernelSmoothedAgent(game, epochs=2_000, lr=0.1, rollout_samples=5, explore_proba=0.1, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = CounterfactualRegretMinimizationAgent(game, epochs=100)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = BatchKernelSmoothedAgent(game, epochs=500, lr=0.01, rollout_samples=1, explore_proba=0.01, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    agent = BatchKernelSmoothedMethodicalAgent(game, epochs=2000, lr=0.01, rollout_samples=1, explore_proba=0.01, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    agents.append(agent)

    # todo
    # agent = BatchKernelSmoothedMethodicalAgent(game, epochs=20_000, lr=0.001, rollout_samples=3, explore_proba=0.01, max_sigma=5.)
    # trainer = AgentTrainer(agent)
    # trainer.run()
    # agents.append(agent)

    # agent = GrokAgent2(game, epochs=100, lr=0.1, rollout_samples=5, explore_proba=0.01, max_sigma=5.)
    # trainer = AgentTrainer(agent)
    # trainer.run()
    # agents.append(agent)

    # agent = NewAgent(game, epochs=30, lr=0.1, rollout_samples=3, explore_proba=0.01, max_sigma=5.)
    # trainer = AgentTrainer(agent, force_training=True)
    # trainer.run()
    # agents.append(agent)

    good_agent = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(good_agent)
    trainer.run()

    new_agent = NewAgent(game, epochs=300, lr=0.01, rollout_samples=5, explore_proba=0.1, max_sigma=0.)
    new_agent.logits = good_agent.logits.copy()
    new_agent.policy = good_agent.policy.copy()
    new_agent.train()
    new_agent.save(DATA_DIR / f"{new_agent}.json")
    agents.append(new_agent)

    new_agent = NewAgent(game, epochs=3_000, lr=0.01, rollout_samples=5, explore_proba=0.01, max_sigma=0.)
    new_agent.logits = good_agent.logits.copy()
    new_agent.policy = good_agent.policy.copy()
    new_agent.train()
    new_agent.save(DATA_DIR / f"{new_agent}.json")
    agents.append(new_agent)

    # Evaluation
    all_v_all_tournament(game, agents, n_games=n_games)


if __name__ == '__main__':
    main()
