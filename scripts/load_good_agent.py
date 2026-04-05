from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.greedy_agent import GreedyAgent
from mini_poker.agents.cfr_agent import CFRAgent


def load_good_expectation_maximization_agent_4_52():
    """CounterfactualEMAgent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                  explore_proba=1., kernel_size=1.)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_crm_agent_4_52():
    """CRMAgent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CRMAgent(game, epochs=4000)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_expectation_maximization_agent_5_52():
    """CounterfactualEMAgent with proven performance."""
    game = MiniPoker(5, 52)

    # Load agent
    # agent = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=10,
    #                               explore_proba=1., kernel_size=2.)
    agent = CounterfactualEMAgent(game, epochs=500, lr=0.003, rollout_samples=3,
                                  explore_proba=1., kernel_size=0.)

    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_crm_agent_5_52():
    """CRMAgent with proven performance."""
    game = MiniPoker(5, 52)

    # Load agent
    agent = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    # agent = CRMAgent(game, epochs=5_000, explore_proba=0.1)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_greedy_agent(n_visits=10, k_greed=5.):
    game = MiniPoker(5, 52)
    balanced_agent = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    greedy_agent = GreedyAgent(balanced_agent, n_visits=n_visits, k_greed=k_greed)
    return greedy_agent


def load_good_search_cfr_agent(n_visits=1000):
    """
    Load a CFR agent with online search enabled.

    :param n_visits: Number of search iterations per decision (default: 1000)
    :return: CFRAgent with search enabled
    """
    game = MiniPoker(5, 52)

    # Load WITHOUT search first (to match saved filename)
    # agent = CFRAgent(game, epochs=1500, search_enabled=False)
    agent = CFRAgent(game, epochs=3000, search_enabled=False)
    agent.load()  # Loads CFRAgent(5,52)_e2000.json

    # Enable search AFTER loading (doesn't affect filename)
    agent.set_search_enabled(True, iterations=n_visits)

    return agent


def load_good_agent(game_power, deck_size):
    """Agent with proven performance."""

    if (game_power, deck_size) == (4, 52):
        # return load_good_expectation_maximization_agent_4_52()
        return load_good_crm_agent_4_52()

    elif (game_power, deck_size) == (5, 52):
        # return load_good_expectation_maximization_agent_5_52()
        # return load_greedy_agent()
        # return load_good_crm_agent_5_52()
        return load_good_search_cfr_agent()

    raise NotImplementedError
