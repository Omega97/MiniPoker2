from mini_poker.game import MiniPoker
from mini_poker.agents.cfr_agent import CFRAgent
from mini_poker.training.evaluation import evaluate_agents


def test_1():
    game = MiniPoker(5, 52)

    agent_1 = CFRAgent(game, epochs=1000, explore_proba=0.1)
    agent_1.load()

    agent_2 = CFRAgent(game, epochs=100, explore_proba=0.1)
    agent_2.load()

    total_epochs = 0
    n_epochs = 1
    avg = 0.
    while True:
        avg_a, avg_b, n_games = evaluate_agents(game, [agent_1, agent_2], epochs=n_epochs)
        avg = (avg * total_epochs + avg_a * n_epochs) / (total_epochs + n_epochs)
        total_epochs += n_epochs
        n_epochs += 1
        print(f"{avg:+.3f}")


def test_2():
    game = MiniPoker(5, 52)

    agent = CFRAgent(game, epochs=300, explore_proba=0.5, kernel_size=0.1)
    agent.load()

    print(agent.show_average_reward())


if __name__ == '__main__':
    # test_1()
    test_2()
