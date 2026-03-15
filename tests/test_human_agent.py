from mini_poker.game import MiniPoker
from mini_poker.agents.counterfactual_agent import CounterfactualAgent
from mini_poker.agents.human_agent import HumanAgent, play_vs_agent


def main():
    # Initialize game with settings from your tests [cite: 83]
    game = MiniPoker(game_power=4, deck_size=52)

    # Load your trained model
    ai_agent = CounterfactualAgent(game,
                                   logit_bound=10.,
                                   epochs=20_000,
                                   lr=0.001,
                                   rollout_samples=2,
                                   explore_proba=0.,)
    try:
        ai_agent.load()  # Uses default path logic [cite: 24]
        print(ai_agent.show_policy())
    except FileNotFoundError:
        print("Trained agent not found, using an untrained one.")

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
