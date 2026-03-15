from mini_poker.game import MiniPoker
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.training.trainer import AgentTrainer


def main():
    # Setup game
    game = MiniPoker(game_power=4,
                     deck_size=52)

    # Setup Agent (modify parameters here)
    agent = CounterfactualAgent(game,
                                epochs=10_000,
                                lr=0.001,
                                rollout_samples=2,
                                explore_proba=.01)

    # Training
    trainer = AgentTrainer(agent, show_policy=True)
    trainer.run()


if __name__ == '__main__':
    main()
