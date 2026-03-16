from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.new_agent import NewAgent


def test_1():
    game = MiniPoker(4, 52)
    agent = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy())


def test_2():
    game = MiniPoker(4, 52)
    agent = NewAgent(game, epochs=300, lr=0.01, rollout_samples=5, explore_proba=0.1, max_sigma=0.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy())


if __name__ == '__main__':
    # test_1()
    test_2()
