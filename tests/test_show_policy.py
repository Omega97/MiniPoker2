from mini_poker.game import MiniPoker
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.training.trainer import AgentTrainer


def main():
    game = MiniPoker(5, 52)

    agent = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=1,
                                         explore_proba=1., max_sigma=2.)
    trainer = AgentTrainer(agent)
    trainer.run()

    print(agent.show_policy())
    agent.sanity_check()


if __name__ == '__main__':
    main()
