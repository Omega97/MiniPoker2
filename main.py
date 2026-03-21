"""
Play 4-52 Mini Poker with AI, or ask the AI the best move

Suits: ♠️s ♣️c ♦️d ♥️h
Ranks: 2 3 4 5 6 7 8 9 T J Q K A
Moves: F C R D T A
"""
from scripts.human_vs_ai import human_vs_ai
from scripts.show_good_policy import show_good_policy
from scripts.ask_the_policy import ask_the_policy


if __name__ == '__main__':

    # === Play with AI ===
    # human_vs_ai()

    # === Show Policy ===
    # show_good_policy()

    # === Ask AI the best move ===
    ask_the_policy("Th", "R")
