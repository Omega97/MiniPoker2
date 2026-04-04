import random
import json
import matplotlib.pyplot as plt
import numpy as np
from mini_poker.paths import GAME_DATA_DIR
from mini_poker.agents.base_agent import BaseAgent, Infoset
from mini_poker.game import State, MiniPoker


# --- AGENT DEFINITION ---

class HumanAgent(BaseAgent):
    """
    An agent that allows a human player to choose actions via the console.
    """

    def __init__(self, game):
        super().__init__(game)

    def get_action(self, infoset: Infoset):
        """Manual input logic for the human player."""
        card, history = infoset.get_values()
        legal_actions = self.game.tree[history]

        print(f"\n--- YOUR TURN ---")
        print(f"✋ Hand: [ {card} ] | History: '{history or '(root)'}'")
        print(f"Legal Actions: {', '.join(legal_actions)}")

        action = ""
        while action not in legal_actions:
            action = input(f"Choose action: ").strip().upper()
        return action

    def __str__(self):
        return "Human"


# --- GAME ENGINE ---

def play_vs_agent(game: MiniPoker, players: list) -> tuple[tuple[float, float], State]:
    """
    Simulates a single game between two agents and returns the rewards and state.
    """
    # 1. Setup: Sample 2 cards from the deck (without replacement)
    cards = random.sample(range(game.deck_size), 2)
    p1_card, p2_card = cards
    agent_p1, agent_p2 = players
    history = ""

    print("\n" + "=" * 30)
    print(f"GAME START: {agent_p1} vs {agent_p2}")

    # 2. Main Game Loop
    while history not in game.terminals:
        acting_player_idx = len(history) % 2
        current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
        current_card = p1_card if acting_player_idx == 0 else p2_card

        infoset = Infoset(current_card, history)

        if not isinstance(current_agent, HumanAgent):
            print(f"\n{current_agent} is thinking...")

        action = current_agent.get_action(infoset)
        history += action
        print(f"Player {acting_player_idx + 1} chose: {action}")

        # Pause for readability if AI is playing
        if not isinstance(current_agent, HumanAgent):
            input("Press Enter to see result...")

    # 3. Wrap up results
    state = State(p1_card, p2_card, branch=history)
    rewards = game.get_reward(state)

    print(f"\nTERMINAL STATE: '{history}'")
    print(f"🟪 P1 Card: [{p1_card:2}] | P2 Card: [{p2_card:2}]")
    print(f"REWARDS -> P1: {rewards[0]:+.1f}, P2: {rewards[1]:+.1f}")
    print("=" * 30)

    return rewards, state


# --- SESSION MANAGER ---

class PlayVsAI:
    def __init__(self, ai_agent: BaseAgent):
        self.ai_agent = ai_agent
        self.game = ai_agent.game
        self.human = HumanAgent(self.game)

        # File naming based on game complexity
        self.save_path = GAME_DATA_DIR / f"detailed_match_history_{self.game.game_power}_{self.game.deck_size}.json"
        self.match_history = self._load_history()

    def _load_history(self):
        """Loads detailed game logs from disk if they exist."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print("Warning: Log file corrupted. Starting fresh.")
        return []

    def save_history(self):
        """Saves session logs to JSON."""
        with open(self.save_path, 'w') as f:
            json.dump(self.match_history, f, indent=4)
        print(f"Data saved to {self.save_path}")

    def play(self):
        """Main loop for playing multiple hands."""
        try:
            while True:
                # Alternate turns: Human first on even hands, AI first on odd
                is_human_first = len(self.match_history) % 2 == 0
                players = [self.human, self.ai_agent] if is_human_first else [self.ai_agent, self.human]

                # Play the actual hand
                rewards, state = play_vs_agent(self.game, players)

                # Store comprehensive data: Names, Cards, Moves, and Outcomes
                entry = {
                    "p1_name": str(players[0]),
                    "p2_name": str(players[1]),
                    "p1_card": state.get_cards()[0],
                    "p2_card": state.get_cards()[1],
                    "moves": state.branch,
                    "results": rewards
                }
                self.match_history.append(entry)

                # Display quick stats for the current hand
                r_human = rewards[0] if players[0] == self.human else rewards[1]
                self._display_quick_stats(r_human, is_human_first)

                user_input = input("Press Enter to continue (or 'q' to quit): ").strip().lower()
                if user_input in ['q', 'quit']:
                    break

        except KeyboardInterrupt:
            print("\nSession interrupted.")
        finally:
            self.save_history()
            self.display_history()  # Show summary table
            self.plot_performance()  # Show graph

    def _display_quick_stats(self, r_human, is_human_first):
        """Calculates and prints performance metrics for the current session."""
        human_rewards = [
            h['results'][0] if h['p1_name'] == "Human" else h['results'][1]
            for h in self.match_history
        ]
        hands = len(human_rewards)
        avg = np.mean(human_rewards)
        std_err = np.std(human_rewards) / (hands ** 0.5) if hands > 1 else 0

        emoji = "🔵" if r_human >= 0 else "🔴"
        starter = "Human" if is_human_first else "AI"
        print(f"\n{emoji} [Hand {hands}] {starter} went first.")
        print(f"Reward: {r_human:+.0f} | Avg: {avg:+.3f} (±{std_err:.3f})")

    def display_history(self):
        """Prints a detailed table of all games played."""
        if not self.match_history:
            return

        print(f"\n{'#' * 20} SESSION LOG {'#' * 20}")
        header = f"{'Hand':<5} | {'P1 (Card)':<15} | {'P2 (Card)':<15} | {'Moves':<10} | {'Human'}"
        print(header)
        print("-" * len(header))

        for i, entry in enumerate(self.match_history, 1):
            is_human_p1 = entry['p1_name'] == "Human"
            human_score = entry['results'][0] if is_human_p1 else entry['results'][1]

            p1_str = f"{entry['p1_name']} ({entry.get('p1_card', '?')})"
            p2_str = f"{entry['p2_name']} ({entry.get('p2_card', '?')})"

            print(f"{i:<5} | {p1_str:<15} | {p2_str:<15} | {entry['moves']:<10} | {human_score:+.1f}")
        print("-" * len(header))

    def plot_performance(self):
        """Visualizes the cumulative reward trend."""
        if not self.match_history: return

        human_rewards = [
            h['results'][0] if h['p1_name'] == "Human" else h['results'][1]
            for h in self.match_history
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum([0] + human_rewards), color='teal', marker='o', markersize=4)
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title(f"Cumulative Performance vs {self.ai_agent}")
        plt.xlabel("Hands Played")
        plt.ylabel("Total Reward")
        plt.grid(True, alpha=0.3)
        plt.show()
