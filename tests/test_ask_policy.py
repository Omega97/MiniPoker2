import numpy as np
from mini_poker.game import MiniPoker
from mini_poker.training.trainer import AgentTrainer
from mini_poker.utils import card_to_num
from mini_poker.agents.base_agent import Infoset
from mini_poker.agents.new_agent import NewAgent
from mini_poker.agents.posterior_sampling_agent import PosteriorSamplingAgent


def load_agent():
    game = MiniPoker(4, 52)

    # Load agent
    # agent = NewAgent(game, epochs=1000, lr=0.001, rollout_samples=20, explore_proba=0.01, max_sigma=2.)
    # agent = PosteriorSamplingAgent(game, epochs=300, lr=0.01, rollout_samples=1, explore_proba=0.01, max_sigma=5.)
    agent = PosteriorSamplingAgent(game, epochs=1_000, lr=0.001, rollout_samples=10, explore_proba=0.01, max_sigma=2.)

    trainer = AgentTrainer(agent)
    trainer.run()
    return agent


def analyze_infoset_and_posterior(agent, my_card, history):
    """
    Inputs:
      agent: The trained agent (contains the policy).
      my_card: The card currently in your hand.
      history: The string of actions (e.g., "bc").

    Returns:
      action_probs: Probability of actions at the CURRENT state.
      posterior: Distribution over what the opponent is representing.
    """
    game = agent.game
    num_cards = game.deck_size

    # 1. Get the current action distribution for OUR hand
    current_infoset = Infoset(my_card, history)
    action_probs = agent.get_policy(current_infoset)

    # 2. Calculate Posterior Distribution over Opponent Cards
    # We look at every card the opponent could have and see how likely
    # they were to play the history given the agent's policy.

    opponent_probs = np.zeros(num_cards)

    for card_opp in range(num_cards):
        if card_opp == my_card:
            opponent_probs[card_opp] = 0.0  # Opponent cannot have my card
            continue

        # Probability that opponent reaches this history with this card
        reach_prob = 1.0
        temp_hist = ""

        # Walk through the history string
        for i, action in enumerate(history):
            # Check whose turn it was at this step
            # Player 0 acts on even indices, Player 1 on odd
            acting_player = i % 2

            # If it was the OPPONENT's turn to act in the past
            # (Assuming you are player 0 or 1 based on current history length)
            my_player_index = len(history) % 2
            if acting_player != my_player_index:
                # How likely was the opponent to take this specific action?
                prev_infoset = Infoset(card_opp, temp_hist)
                probs = agent.get_policy(prev_infoset)
                reach_prob *= probs.get(action, 0.0)

            temp_hist += action

        opponent_probs[card_opp] = reach_prob

    # Normalize to get the posterior (Bayes' Update)
    total_weight = np.sum(opponent_probs)
    if total_weight > 0:
        posterior = opponent_probs / total_weight
    else:
        # If the history is impossible for this policy, return uniform
        posterior = np.ones(num_cards) / (num_cards - 1)
        posterior[my_card] = 0

    return action_probs, posterior


def test_ask_policy(my_hand: str, path: str, length=30):
    """
    Ask the policy what to do
    :param my_hand: "Ah", "Td", "7c", "2s", ...
    :param path: non-terminal game node ("", "C", "CR", "RR", "D", ...)
    :param length:
    :return:
    """
    my_hand_int = card_to_num(*my_hand)

    # Compute proba
    agent = load_agent()
    probs, post = analyze_infoset_and_posterior(agent, my_hand_int, path)

    # 1. Find the maximum probability to use as a denominator for scaling
    max_p = max(post)

    print(f"\nOpponent Hand Likelihood (Relative scale, max={max_p:.1%}):")
    for c, p in enumerate(post):
        # 2. Scale the bar: if max_p is 0.1 and p is 0.1, it becomes 1.0 * length
        # We check if max_p > 0 to avoid division by zero
        bar_len = int((p / max_p) * length) if max_p > 0 else 0
        bar = '=' * bar_len

        # Display the actual percentage alongside the relative bar
        print(f"  Card {c:2}  |{p:6.1%} | {bar}")

    print(f"\nHand: {my_hand_int}")

    print(f"\nMy Action Probs at history '{path}':")
    for action, prob in probs.items():
        print(f"  {action}: {prob:7.1%}")

    actions = list(probs.keys())
    probabilities = list(probs.values())
    recommended_action = np.random.choice(actions, p=probabilities)
    print(f"\nRANDOM ACTION: {recommended_action}")


if __name__ == '__main__':
    test_ask_policy("Ah", "R")
