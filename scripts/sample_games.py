from mini_poker.game import MiniPoker
from scripts.load_good_agent import load_good_agent
from mini_poker.training.game_recorder import (
    generate_game_records,
    print_game_records,
    analyze_game_records,
    print_analysis
)


def main(game_power=5, deck_size=52):
    # Setup game and agents
    game = MiniPoker(game_power=game_power, deck_size=deck_size)

    # Load agent
    agent = load_good_agent()

    # Generate 100 game records
    records = generate_game_records(
        game=game,
        agent_p1=agent,
        agent_p2=agent,
        perspective=0,  # Record P1's reward (CFR agent)
        seed=42,
        verbose=True
    )

    # Print sample records
    print_game_records(records)

    # Analyze results
    analysis = analyze_game_records(records)
    print_analysis(analysis)


if __name__ == '__main__':
    main()
