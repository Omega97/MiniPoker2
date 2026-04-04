from scripts.load_good_agent import load_good_agent
from mini_poker.human_vs_ai import PlayVsAI


def main(game_power=5, deck_size=52):
    ai_agent = load_good_agent(game_power, deck_size)

    # === Play with AI ===
    PlayVsAI(ai_agent=ai_agent).play()


if __name__ == '__main__':
    main()
