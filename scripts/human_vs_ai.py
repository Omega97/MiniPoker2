import matplotlib.pyplot as plt
import numpy as np
import json
from mini_poker.paths import GAME_DATA_DIR
from mini_poker.agents.human_agent import HumanAgent, play_vs_agent


def human_vs_ai(ai_agent):
    # Initialize game settings
    game = ai_agent.game
    game_power = game.game_power
    deck_size = game.deck_size
    save_path = GAME_DATA_DIR / f"match_history_{game_power}_{deck_size}.json"

    # Players
    human = HumanAgent(game)

    # 1. Load existing data if possible
    rewards_history = []
    if save_path.exists():
        with open(save_path, 'r') as f:
            rewards_history = json.load(f)
        print(f"Loaded {len(rewards_history)} previous hands from {save_path}")

    points = sum(rewards_history)
    hands = len(rewards_history)

    try:
        while True:
            # First Hand: Human goes first
            r_human, r_ai = play_vs_agent(game, human, ai_agent)
            rewards_history.append(r_human)
            points += r_human
            hands += 1

            emoji = "🔵" if r_human >= 0 else "🔴"
            std = np.std(rewards_history) / len(rewards_history) ** 0.5
            print(f"{emoji} Your reward = {r_human:+.0f} | Avg: {points / hands:+.3f} (±{std:.3f})")
            user_input = input("\nPress Enter to continue (or 'q' to quit): ").strip().lower()
            if user_input in ['q', 'quit']:
                break

            # Second Hand: AI goes first
            r_ai, r_human = play_vs_agent(game, ai_agent, human)
            rewards_history.append(r_human)
            points += r_human
            hands += 1

            emoji = "🔵" if r_human >= 0 else "🔴"
            std = np.std(rewards_history) / len(rewards_history) ** 0.5
            print(f"{emoji} Your reward = {r_human:+.0f} | Avg: {points / hands:+.3f} (±{std:.3f})")
            user_input = input("\nPress Enter to continue (or 'q' to quit): ").strip().lower()
            if user_input in ['q', 'quit']:
                break

    except KeyboardInterrupt:
        print("\nSession interrupted.")

    # 2. Save all rewards to the file
    with open(save_path, 'w') as f:
        json.dump(rewards_history, f)
    print(f"Data saved to {save_path}")

    # 3. Display cumulative reward with Matplotlib
    if rewards_history:
        cumulative_rewards = [0]
        current_total = 0
        for r in rewards_history:
            current_total += r
            cumulative_rewards.append(current_total)

        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_rewards, label="Cumulative Reward")
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title(f"Human Performance vs {ai_agent}")
        plt.xlabel("Hands Played")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    human_vs_ai()
