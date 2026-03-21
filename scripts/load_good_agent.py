from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent


def load_good_agent():
    """Agent with proven performance."""
    game = MiniPoker(4, 52)

    # Load agent
    agent = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                  explore_proba=1., max_sigma=1.)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent
