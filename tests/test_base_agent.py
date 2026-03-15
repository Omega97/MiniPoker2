from mini_poker.game import MiniPoker
from mini_poker.agents.base_agent import BaseAgent


ROOT = f"..\\mini_poker\\instances\\"


def test_base_agent():
    """Create and show random agent."""
    game = MiniPoker(game_power=2, deck_size=6)
    agent = BaseAgent(game)
    print(agent)
    agent.save(ROOT + f"{agent}.json")


if __name__ == "__main__":
    test_base_agent()
