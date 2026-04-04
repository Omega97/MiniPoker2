from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.crm_agent import CRMAgent


def test_1(game_power=5, deck_size=52):
    game = MiniPoker(game_power, deck_size)
    agent = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    AgentTrainer(agent).run()

    # print(agent.show_policy())
    print(agent.show_average_reward())


if __name__ == '__main__':
    test_1()
