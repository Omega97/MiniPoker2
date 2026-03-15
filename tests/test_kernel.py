from mini_poker.game import MiniPoker
from mini_poker.agents.kernel_smoothed import KernelSmoothedAgent


def main():
    game = MiniPoker(4, 52)
    agent = KernelSmoothedAgent(game, max_sigma=5.)
    agent.print_kernel_weights()


if __name__ == '__main__':
    main()
