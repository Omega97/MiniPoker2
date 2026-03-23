from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent


def load_good_expectation_maximization_agent_4_52():
    """CounterfactualEMAgent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                  explore_proba=1., max_sigma=1.)
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
    agent = CounterfactualEMAgent(game, epochs=300, lr=0.005, rollout_samples=10,
                                  explore_proba=1., max_sigma=2.)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_crm_agent_5_52():
    """CRMAgent with proven performance."""
    game = MiniPoker(5, 52)

    # Load agent
    agent = CRMAgent(game, epochs=500, explore_proba=0.1)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_agent(game_power=4, deck_size=52):
    """Agent with proven performance."""

    if (game_power, deck_size) == (4, 52):
        # return load_good_expectation_maximization_agent_4_52()
        return load_good_crm_agent_4_52()

    elif (game_power, deck_size) == (5, 52):
        return load_good_expectation_maximization_agent_5_52()
        # return load_good_crm_agent_5_52()

    return
