from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.crm_agent import CRMAgent


def load_good_expectation_maximization_agent():
    """CounterfactualEMAgent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                  explore_proba=1., max_sigma=1.)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_crm_agent():
    """CRMAgent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CRMAgent(game, epochs=4000)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def load_good_agent():
    """Agent with proven performance."""
    return load_good_expectation_maximization_agent()
    # return load_good_crm_agent()
