from scripts.load_good_agent import load_good_agent


def show_good_policy():
    agent = load_good_agent()
    print(agent.show_policy())
    agent.sanity_check()


if __name__ == '__main__':
    show_good_policy()
