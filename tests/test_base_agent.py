from mini_poker.game import MiniPoker, State
from mini_poker.paths import DATA_DIR
from mini_poker.agents.base_agent import BaseAgent


def test_base_agent():
    """Create and show random agent."""
    game = MiniPoker(game_power=5, deck_size=52)
    agent = BaseAgent(game)
    print(agent)
    agent.save(DATA_DIR / f"{agent}.json")


def test_2(n_samples=500):
    """Create and show random agent."""
    game = MiniPoker(game_power=2, deck_size=5)

    agent = BaseAgent(game, memory_period=20)
    state = State(1, 2, "")

    with agent.train_context():
        for _ in range(n_samples):
            agent.sample_trajectory(state)

    print(agent.show_average_reward())

    print(f"{len(agent.trajectories_cache)} trajectories")
    print(agent.trajectories_cache[0])


if __name__ == "__main__":
    # test_base_agent()
    test_2()
