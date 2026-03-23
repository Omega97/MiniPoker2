from mini_poker.agents.search_agent import TerminalOptimalEMAgent
from mini_poker.training.evaluation import all_v_all_tournament
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.cem2_agent import CounterfactualEMAgent
from mini_poker.agents.base_agent import BaseAgent
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.counterfactual_agent import CounterfactualAgent


def main():
    game = MiniPoker(4, 52)

    # Load BaseAgent
    base_agent = BaseAgent(game)

    # Load CounterfactualAgent
    agent_good = CounterfactualAgent(game, epochs=20_000, lr=0.001, rollout_samples=2, explore_proba=0.)
    trainer = AgentTrainer(agent_good)
    trainer.run()

    # Load CounterfactualEMAgent
    em_agent = CounterfactualEMAgent(game, epochs=100, lr=0.01, rollout_samples=1,
                                  explore_proba=1., max_sigma=1.)
    trainer = AgentTrainer(em_agent)
    trainer.run()

    # Load TerminalOptimalEMAgent
    search_agent = TerminalOptimalEMAgent(game)
    search_agent.inherit_from(em_agent)

    # Load CRM
    crm_agent = CRMAgent(game, epochs=4000, explore_proba=.1)
    trainer = AgentTrainer(crm_agent)
    trainer.run()

    agents = [base_agent, search_agent, em_agent, crm_agent, agent_good]
    all_v_all_tournament(game, agents, n_games=50_000)

    # print(search_agent.show_policy())


if __name__ == '__main__':
    main()
