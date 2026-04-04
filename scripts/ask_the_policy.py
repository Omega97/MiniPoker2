import numpy as np
from mini_poker.utils import card_to_num, clip_proba
from mini_poker.agents.base_agent import Infoset
from mini_poker.utils import print_colored_status
from scripts.load_good_agent import load_good_agent


def ask_the_policy(my_hand: str | int, branch: str, agent,
                   length=30, p_threshold=1e-3):
    """
    Ask the policy what to do, with Bayesian EV estimation.

    :param my_hand: "Ah", "Td", "7c", ..., "2s" or 0, 1, ..., 51
    :param branch: non-terminal game node ("", "C", "CR", "RR", "D", ...)
    :param agent: trained BaseAgent instance
    :param length: width of posterior likelihood bar chart
    :param p_threshold: probability threshold for clipping low-prob actions
    :return: None (prints analysis to console)
    """
    # Convert hand to integer index
    my_hand_int = card_to_num(*my_hand) if isinstance(my_hand, str) else my_hand
    infoset = Infoset(my_hand_int, branch)

    # Compute action probabilities and posterior over opponent's card
    probs, post = agent.analyze_infoset_and_posterior(infoset, exclude_self_card=True)

    # Display opponent hand likelihood (relative scale)
    max_p = max(post) if np.any(post) else 1.0
    print(f"\nOpponent Hand Likelihood (Relative scale, max={max_p:.1%}):")
    for c, p in enumerate(post):
        bar_len = int((p / max_p) * length) if max_p > 0 else 0
        bar = '=' * bar_len
        print(f"  Card {c:2}  |{p:6.1%} | {bar}")

    # Prepare action probabilities with clipping
    actions = list(probs.keys())
    probabilities = np.array(list(probs.values()))
    probabilities = clip_proba(probabilities, threshold=p_threshold)
    recommended_action = np.random.choice(actions, p=probabilities)

    print(f"\nHand: {my_hand_int}")
    print(f"\nMy Action Probs at history '{branch}':")

    for action in actions:
        prob = agent.get_policy(infoset)[action]
        ev = agent.get_average_rewards(infoset)[action]
        s_ev = print_colored_status(ev, text=f"{ev:+.2f}")
        print(f"  {action}: {prob:7.1%}     EV = {s_ev}")

    print(f"\n→ RANDOM ACTION: {recommended_action}")


def main():

    # === Load agent ===
    agent = load_good_agent(game_power=5, deck_size=52)

    # === Ask the policy ===
    # ask_the_policy(my_hand=2, branch="DD", agent=agent)
    # ask_the_policy(my_hand=9, branch="DD", agent=agent)
    # ask_the_policy(my_hand=10, branch="CR", agent=agent)
    # ask_the_policy(my_hand=12, branch="RR", agent=agent)
    # ask_the_policy(my_hand=14, branch="CD", agent=agent)
    # ask_the_policy(my_hand=17, branch="CD", agent=agent)
    # ask_the_policy(my_hand=20, branch="RD", agent=agent)
    # ask_the_policy(my_hand=25, branch="CR", agent=agent)
    # ask_the_policy(my_hand=26, branch="CD", agent=agent)
    # ask_the_policy(my_hand=29, branch="R", agent=agent)
    # ask_the_policy(my_hand=30, branch="R", agent=agent)
    # ask_the_policy(my_hand=34, branch="RD", agent=agent)
    # ask_the_policy(my_hand=35, branch="CRA", agent=agent)
    # ask_the_policy(my_hand=38, branch="CRR", agent=agent)
    # ask_the_policy(my_hand=41, branch="RR", agent=agent)
    # ask_the_policy(my_hand=42, branch="CT", agent=agent)
    # ask_the_policy(my_hand=43, branch="RR", agent=agent)
    # ask_the_policy(my_hand=46, branch="RD", agent=agent)
    # ask_the_policy(my_hand=48, branch="RA", agent=agent)
    # ask_the_policy(my_hand=50, branch="CRR", agent=agent)
    ask_the_policy(my_hand=51, branch="RR", agent=agent)


if __name__ == '__main__':
    main()
