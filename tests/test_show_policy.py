from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.batch_kernel_smoothed_methodical_agent import BatchKernelSmoothedMethodicalAgent
from mini_poker.agents.new_agent import NewAgent
from mini_poker.agents.posterior_sampling_agent import PosteriorSamplingAgent


def test_1():
    game = MiniPoker(4, 52)
    agent = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy())


def test_2():
    game = MiniPoker(4, 52)
    agent = NewAgent(game, epochs=1000, lr=0.001, rollout_samples=20, explore_proba=0.01, max_sigma=2.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy(logit_mode=False))
    agent.sanity_check()


def test_3():
    game = MiniPoker(4, 52)
    agent = BatchKernelSmoothedMethodicalAgent(game, epochs=2000, lr=0.01, rollout_samples=1, explore_proba=0.01, max_sigma=5.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy())
    agent.sanity_check()


def test_show_good_policy():
    game = MiniPoker(4, 52)
    agent = PosteriorSamplingAgent(game, epochs=1_000, lr=0.001, rollout_samples=10, explore_proba=0.01, max_sigma=2.)
    trainer = AgentTrainer(agent)
    trainer.run()
    print(agent.show_policy())
    agent.sanity_check()


if __name__ == '__main__':
    # test_1()
    # test_2()
    # test_3()
    test_show_good_policy()
