from scripts.load_good_agent import load_good_agent


def show_good_policy(agent):
    policy = agent.show_policy()
    print(policy)
    agent.sanity_check()


if __name__ == '__main__':
    agent = load_good_agent(5, 52)
    show_good_policy(agent)
