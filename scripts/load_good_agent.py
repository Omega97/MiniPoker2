from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.decisive_agent import DecisiveAgent
from mini_poker.agents.greedy_agent import GreedyAgent


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
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_decisive_agent_5_52():
    game = MiniPoker(5, 52)

    agent_crm = CRMAgent(game, epochs=5000, explore_proba=0.1)
    trainer = AgentTrainer(agent_crm)
    trainer.run()

    agent = DecisiveAgent(game, epsilon=1e-3)
    agent.inherit_from(agent_crm)

    return agent


def load_greedy_agent(n_visits=10, k_greed=5.):
    game = MiniPoker(5, 52)

    balanced_agent = CRMAgent(game, epochs=20_000, explore_proba=0.01)
    greedy_agent = GreedyAgent(balanced_agent, n_visits=n_visits, k_greed=k_greed)

    return greedy_agent


def load_good_agent(game_power, deck_size):
    """Agent with proven performance."""

    if (game_power, deck_size) == (4, 52):
        # return load_good_expectation_maximization_agent_4_52()
        return load_good_crm_agent_4_52()

    elif (game_power, deck_size) == (5, 52):
        # return load_good_expectation_maximization_agent_5_52()
        # return load_good_decisive_agent_5_52()
        return load_good_crm_agent_5_52()
        # return load_greedy_agent()

    raise NotImplementedError
