import random
import numpy as np
from typing import List, Tuple, Optional
from mini_poker.game import MiniPoker, Infoset, State
from mini_poker.agents.base_agent import BaseAgent


def generate_game_records(
        game: MiniPoker,
        agent_p1: BaseAgent,
        agent_p2: BaseAgent,
        perspective: int = 0,  # 0 for P1's reward, 1 for P2's reward
        seed: Optional[int] = None,
        verbose: bool = False,
) -> List[Tuple[int, int, str, float]]:
    """
    Generate game records between two agents.

    Args:
        game: MiniPoker game instance
        agent_p1: Agent playing as Player 1
        agent_p2: Agent playing as Player 2
        perspective: Which player's reward to record (0 or 1)
        seed: Random seed for reproducibility
        verbose: Print progress during generation

    Returns:
        List of tuples: (card_p1, card_p2, action_history, reward)
    """
    if seed is not None:
        random.seed(seed)

    records = []
    games_played = 0

    for p1_card, p2_card in game.iter_uniformly_over_hands(n_sweeps=1):

        # Play the game
        history = ""
        while history not in game.terminals:
            acting_player_idx = len(history) % 2
            current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
            current_card = p1_card if acting_player_idx == 0 else p2_card

            infoset = Infoset(current_card, history)
            action = current_agent.get_action(infoset)
            history += action

        # Calculate rewards
        state = State(p1_card, p2_card, branch=history)
        r1, r2 = game.get_reward(state)
        reward = r1 if perspective == 0 else r2

        records.append((p1_card, p2_card, history, reward))
        games_played += 1

    return records


def print_game_records(records: List[Tuple[int, int, str, float]]):
    """
    Print game records in a formatted way.

    Args:
        records: List of game records from generate_game_records()
    """
    print(f"\n{'Card P1':<8} {'Card P2':<8} {'History':<12} {'Reward':<8}")
    print("-" * 40)

    for i, (card_p1, card_p2, history, reward) in enumerate(records):
        reward_str = f"{reward:+3.0f}" if reward == int(reward) else f"{reward:+.1f}"
        print(f"{card_p1:<8} {card_p2:<8} {history:<12} {reward_str:<8}")


def save_game_records(records: List[Tuple[int, int, str, float]], filepath: str):
    """
    Save game records to a text file.

    Args:
        records: List of game records
        filepath: Path to save the records
    """
    with open(filepath, 'w') as f:
        for card_p1, card_p2, history, reward in records:
            reward_str = f"{reward:+.0f}" if reward == int(reward) else f"{reward:+.1f}"
            f.write(f"{card_p1}, {card_p2}, \"{history}\", {reward_str}\n")
    print(f"Saved {len(records)} game records to {filepath}")


def load_game_records(filepath: str) -> List[Tuple[int, int, str, float]]:
    """
    Load game records from a text file.

    Args:
        filepath: Path to load the records from

    Returns:
        List of game records
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse: 2, 40, "CRF", -1
            parts = line.split(', ')
            card_p1 = int(parts[0])
            card_p2 = int(parts[1])
            history = parts[2].strip('"')
            reward = float(parts[3])
            records.append((card_p1, card_p2, history, reward))
    return records


def analyze_game_records(records: List[Tuple[int, int, str, float]]) -> dict:
    """
    Analyze game records for statistics.

    Args:
        records: List of game records

    Returns:
        Dictionary with analysis statistics
    """
    if not records:
        return {}

    rewards = [r for _, _, _, r in records]
    histories = [h for _, _, h, _ in records]

    # Calculate statistics
    avg_reward = sum(rewards) / len(rewards)
    win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    loss_rate = sum(1 for r in rewards if r < 0) / len(rewards)
    tie_rate = sum(1 for r in rewards if r == 0) / len(rewards)

    # Most common action sequences
    from collections import Counter
    history_counts = Counter(histories)
    most_common_histories = history_counts.most_common(5)

    # Average hand strength for wins vs losses
    win_cards_p1 = [c1 for c1, _, _, r in records if r > 0]
    loss_cards_p1 = [c1 for c1, _, _, r in records if r < 0]
    avg_win_card = sum(win_cards_p1) / len(win_cards_p1) if win_cards_p1 else 0
    avg_loss_card = sum(loss_cards_p1) / len(loss_cards_p1) if loss_cards_p1 else 0

    reward_std = np.std(rewards)

    return {
        'total_games': len(records),
        'avg_reward': avg_reward,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'tie_rate': tie_rate,
        'most_common_histories': most_common_histories,
        'avg_win_card': avg_win_card,
        'avg_loss_card': avg_loss_card,
        'reward_std': reward_std,
    }


def print_analysis(analysis: dict):
    """Print analysis results in a formatted way."""
    print("\n" + "=" * 50)
    print("GAME RECORD ANALYSIS")
    print("=" * 50)
    print(f"Total Games:     {analysis['total_games']:6}")
    print(f"Average Reward:  {analysis['avg_reward']:+7.2f}")
    print(f"Win Rate:        {analysis['win_rate']:7.1%}")
    print(f"Loss Rate:       {analysis['loss_rate']:7.1%}")
    print(f"Avg Card (Win):  {analysis['avg_win_card']:6.1f}")
    print(f"Avg Card (Loss): {analysis['avg_loss_card']:6.1f}")
    print(f"Reward std:      {analysis['reward_std']:7.2f}")
    print("\nMost Common Action Sequences:")
    for history, count in analysis['most_common_histories']:
        print(f"  {history:<10} {count:5} times")
    print("=" * 50)
