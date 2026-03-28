import numpy as np
from mini_poker.utils import card_to_num, clip_proba
from mini_poker.agents.base_agent import Infoset
from scripts.load_good_agent import load_good_agent


def ask_the_policy(my_hand: str | int, branch: str, agent,
                   length=30, p_threshold=1e-3,
                   n_posterior_samples=100):
    """
    Ask the policy what to do, with Bayesian EV estimation.

    :param my_hand: "Ah", "Td", "7c", ..., "2s" or 0, 1, ..., 51
    :param branch: non-terminal game node ("", "C", "CR", "RR", "D", ...)
    :param agent: trained BaseAgent instance
    :param length: width of posterior likelihood bar chart
    :param p_threshold: probability threshold for clipping low-prob actions
    :param n_posterior_samples: number of opponent cards to sample for EV estimation
    :return: None (prints analysis to console)
    """
    # Convert hand to integer index
    my_hand_int = card_to_num(*my_hand) if isinstance(my_hand, str) else my_hand
    infoset = Infoset(my_hand_int, branch)

    # Compute action probabilities and posterior over opponent's card
    # FIX: exclude_self_card=True ensures opponent can't hold our card
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

    # Determine which player is acting (for EV indexing)
    acting_player_idx = len(branch) % 2  # 0 = P1, 1 = P2

    print(f"\nHand: {my_hand_int}")
    print(f"\nMy Action Probs at history '{branch}':")

    for action, prob in zip(actions, probabilities):
        evs = agent.bayesian_evaluate_action(
            infoset,
            action,
            n_samples=n_posterior_samples,  # posterior samples
        )
        ev = evs[acting_player_idx]  # Select EV for the acting player
        print(f"  {action}: {prob:7.1%}     EV = {ev:+.2f}")

    print(f"\n→ RECOMMENDED ACTION: {recommended_action}")


def main():

    # Load agent
    agent = load_good_agent(game_power=5, deck_size=52)

    # Ask the policy
    # ask_the_policy(21, "CT", agent=agent)
    ask_the_policy(28, "CRT", agent=agent, n_posterior_samples=1000)


if __name__ == '__main__':
    main()
