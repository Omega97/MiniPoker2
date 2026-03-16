from mini_poker.game import MiniPoker
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.human_agent import HumanAgent, play_vs_agent
from mini_poker.training.trainer import AgentTrainer
from mini_poker.agents.new_agent import NewAgent


def load_agent():
    game = MiniPoker(4, 52)
    agent = NewAgent(game, epochs=300, lr=0.01, rollout_samples=5, explore_proba=0.1, max_sigma=0.)
    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def main():
    # Initialize game with settings from your tests [cite: 83]
    game = MiniPoker(game_power=4, deck_size=52)

    # Players
    ai_agent = load_agent()
    human = HumanAgent(game)

    # Play a match!
    points = 0
    hands = 0
    while True:
        r_human, r_ai = play_vs_agent(game, human, ai_agent)
        points += r_human

        # Swap human and ai_agent to change who goes first.
        r_ai, r_human = play_vs_agent(game, ai_agent, human)
        points += r_human

        hands += 2
        print(f"Avg points: {points/hands:+.3f}")


if __name__ == "__main__":
    main()
