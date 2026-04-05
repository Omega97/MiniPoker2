"""
Play 4-52 Mini Poker with AI, or ask the AI the best move

Suits: ♠️s ♣️c ♦️d ♥️h
Ranks: 2 3 4 5 6 7 8 9 T J Q K A
Moves: F C R D T A
"""
from scripts.load_good_agent import load_good_agent
from scripts.play_with_ai import PlayVsAI
from scripts.show_good_policy import show_good_policy
from scripts.ask_the_policy import ask_the_policy


def main(game_power=5, deck_size=52):
    ai_agent = load_good_agent(game_power, deck_size)

    # === Play with AI ===
    # PlayVsAI(ai_agent=ai_agent).play()
    # PlayVsAI(ai_agent=ai_agent).display_history()

    # === Show Policy ===
    print(ai_agent.show_policy())
    # print(ai_agent.show_average_reward())

    # === Ask AI the best move ===
    # ask_the_policy(40, "RRR", agent=ai_agent)


if __name__ == '__main__':
    main()
