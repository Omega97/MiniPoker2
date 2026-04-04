from mini_poker.game import MiniPoker
from mini_poker.agents.cem2_agent import CounterfactualEMAgent


def test_1():
    game = MiniPoker(5, 52)

    agent = CounterfactualEMAgent(game, epochs=300, explore_proba=0.5, kernel_size=2.)

    agent.print_kernel_weights()


if __name__ == '__main__':
    test_1()
