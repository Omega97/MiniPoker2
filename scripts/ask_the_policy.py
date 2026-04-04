from scripts.load_good_agent import load_good_agent
from mini_poker.ask_policy import ask_the_policy


def main(game_power=5, deck_size=52):

    # --- Load agent ---
    ai_agent = load_good_agent(game_power, deck_size)

    # --- Ask agent ---
    ask_the_policy(
        my_hand=48,     # <- hand
        branch="RR",     # <- branch
        agent=ai_agent)


if __name__ == '__main__':
    main()
