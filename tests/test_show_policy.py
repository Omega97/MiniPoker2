from mini_poker.game import MiniPoker
from mini_poker.agents.crm_agent import CRMAgent


def main():
    game = MiniPoker(4, 52)
    agent = CRMAgent(game, epochs=300)
    agent.load()
    print(agent.show_policy())
    agent.sanity_check()


if __name__ == '__main__':
    main()
