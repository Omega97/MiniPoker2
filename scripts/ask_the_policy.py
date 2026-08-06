"""
Ask AI what move to play.

Suits: ♠️s ♣️c ♦️d ♥️h
Ranks: 2 3 4 5 6 7 8 9 T J Q K A
Moves: F C R D T Q A
"""
from scripts.load_good_agent import load_good_agent
from mini_poker.ask_policy import ask_the_policy


def main(game_power=5, deck_size=52):

    # --- Load agent ---
    ai_agent = load_good_agent(game_power, deck_size)

    # --- Your Position ---
    # my_hand  ("Th", 35, ...)
    # branch  ("", "C", "RD", "TA", "Q", ...)
    my_hand = "Th"
    branch = ""

    # --- Ask agent ---
    ask_the_policy(my_hand, branch, agent=ai_agent)


if __name__ == '__main__':
    main()
