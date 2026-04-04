import matplotlib.pyplot as plt
import numpy as np
import json
from mini_poker.paths import GAME_DATA_DIR
from mini_poker.utils import print_colored_status
from mini_poker.agents.human_agent import HumanAgent, play_vs_agent
from mini_poker.agents.base_agent import BaseAgent


class PlayVsAI:

    def __init__(self, ai_agent: BaseAgent):
        self.ai_agent = ai_agent
        self.game = ai_agent.game
        self.human = HumanAgent(self.game)

        # Define the file path based on game settings
        self.save_path = GAME_DATA_DIR / f"detailed_match_history_{self.game.game_power}_{self.game.deck_size}.json"

        # Load existing history or initialize new list
        self.match_history = self._load_history()

    def _load_history(self):
        """Loads detailed game logs from disk."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} previous hands from {self.save_path}")
                return data
            except json.JSONDecodeError:
                print("Warning: Log file corrupted. Starting fresh.")
        return []

    def save_history(self):
        """Saves the current match history to a JSON file."""
        with open(self.save_path, 'w') as f:
            json.dump(self.match_history, f, indent=4)
        print(f"Data saved to {self.save_path}")

    def play(self):
        """Starts the interactive game loop."""
        try:
            while True:
                # Alternate who acts first
                is_human_first = len(self.match_history) % 2 == 0
                players = [self.human, self.ai_agent] if is_human_first else [self.ai_agent, self.human]

                # Play the hand
                # play_vs_agent returns (rewards, state) where state contains p1_card and p2_card
                rewards, state = play_vs_agent(self.game, players)

                # Record detailed log entry including hands (cards)
                entry = {
                    "p1_name": str(players[0]),
                    "p2_name": str(players[1]),
                    "p1_hand": state.get_cards()[0],
                    "p2_hand": state.get_cards()[1],
                    "moves": state.branch,
                    "results": rewards
                }
                self.match_history.append(entry)

                # Extract stats specifically for the human
                r_human = rewards[0] if players[0] == self.human else rewards[1]
                self._display_stats(r_human, is_human_first)

                user_input = input("Press Enter to continue (or 'q' to quit): ").strip().lower()
                if user_input in ['q', 'quit']:
                    break

        except KeyboardInterrupt:
            print("\nSession interrupted.")
        finally:
            self.save_history()
            self.display_history()
            self.plot_performance()

    def _display_stats(self, r_human, is_human_first):
        """Calculates and prints real-time performance metrics."""
        human_rewards = [
            h['results'][0] if h['p1_name'] == str(self.human) else h['results'][1]
            for h in self.match_history
        ]

        hands_count = len(human_rewards)
        avg = np.mean(human_rewards)
        std_err = np.std(human_rewards) / (hands_count ** 0.5) if hands_count > 1 else 0

        emoji = "🔵" if r_human >= 0 else "🔴"
        starter = "Human" if is_human_first else "AI"
        print(f"\n{emoji} [Hand {hands_count}] {starter} went first.")
        print(f"Moves: {' -> '.join(map(str, self.match_history[-1]['moves']))}")
        print(f"Reward: {r_human:+.0f} | Avg: {avg:+.3f} (±{std_err:.3f})")

    def plot_performance(self):
        """Visualizes the human's cumulative progress."""
        if not self.match_history:
            return

        human_rewards = [
            h['results'][0] if h['p1_name'] == str(self.human) else h['results'][1]
            for h in self.match_history
        ]

        cumulative_rewards = np.cumsum([0] + human_rewards)
        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_rewards, label="Human Cumulative Reward", color='teal', linewidth=2)
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title(f"Human Performance vs {self.ai_agent}")
        plt.xlabel("Hands Played")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()

    def display_history(self):
        """Prints a line-by-line summary of all matches including cards."""
        if not self.match_history:
            print("No history found to display.")
            return

        print(f"\n{'=' * 15} MATCH HISTORY SUMMARY {'=' * 15}")
        print()

        # Column titles
        hand_num_t = "Hand#"
        hands_cards_t = "Hole cards"
        moves_t = "Sequence"
        score_t = "Score"
        player_t = "First Player"

        # Define widths
        w_num = len(hand_num_t)
        w_cards = len(hands_cards_t)
        w_moves = len(moves_t)
        w_score = len(score_t)

        header = f"{hand_num_t} | {hands_cards_t} | {moves_t:<{w_moves}} | {score_t} | {player_t}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for i, entry in enumerate(self.match_history, 1):
            is_human_p1 = entry['p1_name'] == str(self.human)
            human_score = entry['results'][0] if is_human_p1 else entry['results'][1]

            # Format the cards column (e.g., " 1 | 4 ")
            # Using .get() for p1_hand to support older JSON files that didn't have them
            p1_card = entry.get('p1_hand', '?')
            p2_card = entry.get('p2_hand', '?')
            hands_str = f" ({p1_card:<2}, {p2_card:<2}) "

            moves_str = entry['moves']
            score_str = print_colored_status(human_score, f"{human_score:<+{w_score}.0f}")
            first_player = entry['p1_name']

            print(f"{i:<{w_num}} | {hands_str:<{w_cards}} | {moves_str:<{w_moves}} | {score_str} | {first_player}")

        print("-" * len(header))
