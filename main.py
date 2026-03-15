import os
import random
from mini_poker.game import MiniPoker
from mini_poker.agents.counterfactual_agent import CounterfactualAgent


def test_base_agent(epochs=1000, lr=0.01, rollout_samples=10, random_seed=0):
    """Create and show random agent."""
    random.seed(random_seed)
    game = MiniPoker(game_power=2, deck_size=6)
    agent = CounterfactualAgent(game)

    i = 0
    while True:
        i += 1
        agent.train(epochs=epochs, lr=lr / i, rollout_samples=rollout_samples)

        # Clear the console screen
        # 'nt' is for Windows, 'posix' is for Linux/macOS
        os.system('cls' if os.name == 'nt' else 'clear')
        print(agent)


def main():
    try:
        # You can adjust these parameters as needed
        test_base_agent(epochs=1000, lr=0.01, rollout_samples=3)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")


if __name__ == "__main__":
    main()
