from mini_poker.game import MiniPoker, State
from mini_poker.paths import DATA_DIR
from mini_poker.agents.base_agent import BaseAgent


def test_base_agent():
    """Create and show random agent."""
    game = MiniPoker(game_power=5, deck_size=52)
    agent = BaseAgent(game)
    print(agent)
    agent.save(DATA_DIR / f"{agent}.json")


def test_2(n_sweeps=20):
    """Create and show random agent."""
    game = MiniPoker(game_power=2, deck_size=5)

    agent = BaseAgent(game, memory_period=20)

    for card1, card2 in game.iter_uniformly_over_hands():
        state = State(card1, card2, "")
        with agent.train_context():
            for _ in range(n_sweeps):
                agent.sample_trajectory_from_root(state)

    print(agent.show_average_reward())

    # print(f"{len(agent.trajectories_cache)} trajectories")
    # print(agent.trajectories_cache[0])


if __name__ == "__main__":
    # test_base_agent()
    test_2()
