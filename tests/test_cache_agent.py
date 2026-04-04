from mini_poker.game import MiniPoker
from mini_poker.agents.cached_counterfactual_agent import CachedCounterfactualAgent


def main(game_power=5, deck_size=52):
    game = MiniPoker(game_power, deck_size)
    agent = CachedCounterfactualAgent(game, epochs=2000, lr=0.01)
    agent.load()
    print(agent.show_average_reward())


if __name__ == '__main__':
    main()
