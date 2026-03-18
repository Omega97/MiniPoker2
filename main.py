"""
Play 4-52 Mini Poker with AI, or ask the AI the best move

Suits: ♠️s ♣️c ♦️d ♥️h
Ranks: 2 3 4 5 6 7 8 9 T J Q K A
Moves: F C R D T A
"""
from tests.test_human_agent import test_human_v_ai
from tests.test_ask_policy import test_ask_policy
from tests.test_show_policy import test_show_good_policy

if __name__ == '__main__':

    # === Play with AI ===
    # test_human_v_ai()

    # === Show Policy ===
    test_show_good_policy()

    # === Ask AI the best move ===
    # test_ask_policy("Kh", "RR")

