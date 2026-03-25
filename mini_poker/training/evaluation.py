import numpy as np
import random
from mini_poker.utils import print_colored_status
from mini_poker.game import MiniPoker, Infoset, State


def _evaluate_agents_fixed_position(game: MiniPoker, agent_p1, agent_p2, n_games=1000):
    """
    Plays two agents against each other for n_games.
    Returns the average reward per game for (Player 1, Player 2).
    """
    total_rewards_p1 = []
    total_rewards_p2 = []

    # Iterate through games
    n_combinations = game.deck_size * (game.deck_size - 1)
    epochs = max(1, round(n_games / n_combinations))

    for p1_card, p2_card in game.iter_uniformly_over_hands(epochs):
        # 1. Deal random cards (permutations ensures no duplicate cards)
        history = ""

        # 2. Play the game until a terminal state is reached
        while history not in game.terminals:
            # Determine whose turn it is
            acting_player_idx = len(history) % 2

            # Select the correct agent and card for the current turn
            current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
            current_card = p1_card if acting_player_idx == 0 else p2_card

            # Get action based on the agent's learned policy
            infoset = Infoset(current_card, history)
            action = current_agent.get_action(infoset)
            history += action

        # 3. Calculate rewards from the terminal state
        state = State(p1_card, p2_card, branch=history)
        r1, r2 = game.get_reward(state)
        total_rewards_p1.append(r1)
        total_rewards_p2.append(r2)

    avg_rewards_p1 = np.mean(total_rewards_p1)
    avg_rewards_p2 = np.mean(total_rewards_p2)

    return avg_rewards_p1, avg_rewards_p2


def evaluate_agents(game: MiniPoker, agent_a, agent_b, n_games=10_000):
    """
    Evaluates two agents by swapping positions halfway through to eliminate
    positional bias. Returns the net average for Agent A and Agent B.
    """
    half_n = n_games // 2

    # Leg 1: Agent A is P1, Agent B is P2
    p1_score_leg1, p2_score_leg1 = _evaluate_agents_fixed_position(game, agent_a, agent_b, n_games=half_n)

    # Leg 2: Agent B is P1, Agent A is P2
    p1_score_leg2, p2_score_leg2 = _evaluate_agents_fixed_position(game, agent_b, agent_a, n_games=half_n)

    # Average performance for Agent A
    # (P1 score when A was first + P2 score when A was second) / 2
    avg_a = (p1_score_leg1 + p2_score_leg2) / 2
    avg_b = (p2_score_leg1 + p1_score_leg2) / 2

    return avg_a, avg_b


def all_v_all_tournament(game: MiniPoker, agents: list, n_games=100_000, random_seed=0):
    random.seed(random_seed)
    name_len = max([len(f"{agent}") for agent in agents])
    print(f"\nResults after {n_games} games")
    for i in range(len(agents)):
        agent_1 = agents[i]
        print(f"{str(agent_1):>{name_len}}", end=" ")
        for j in range(i):
            agent_2 = agents[j]
            avg_p1, avg_p2 = evaluate_agents(game, agent_1, agent_2, n_games=n_games)
            s = print_colored_status(round(avg_p1, 2), text=f"{avg_p1:+5.2f}")
            print(f" {s}", end=' ')
        print()


def quick_evaluate_agents(game: MiniPoker, agent_a, agent_b, n_games=10_000):
    """
    Evaluates two agents by sampling random hands and alternating positions
    every game to eliminate positional bias. Much faster than exhaustive
    evaluation for large deck sizes.

    Returns the net average reward for (Agent A, Agent B).
    """
    total_a, total_b = 0.0, 0.0

    for game_idx in range(n_games):
        # Sample two distinct cards uniformly at random
        p1_card, p2_card = random.sample(range(game.deck_size), 2)
        history = ""

        # Alternate positions every game: even=A@P1, odd=B@P1
        if game_idx % 2 == 0:
            agent_p1, agent_p2 = agent_a, agent_b
            card_p1, card_p2 = p1_card, p2_card
        else:
            agent_p1, agent_p2 = agent_b, agent_a
            card_p1, card_p2 = p2_card, p1_card  # Swap cards too!

        # Play until terminal
        while history not in game.terminals:
            acting_player_idx = len(history) % 2
            current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
            current_card = card_p1 if acting_player_idx == 0 else card_p2

            infoset = Infoset(current_card, history)
            action = current_agent.get_action(infoset)
            history += action

        # Accumulate rewards for each agent (not position!)
        state = State(p1_card, p2_card, branch=history)  # Original card assignment
        r1, r2 = game.get_reward(state)

        if game_idx % 2 == 0:
            # A was P1, B was P2
            total_a += r1
            total_b += r2
        else:
            # B was P1, A was P2
            total_b += r1
            total_a += r2

    return total_a / n_games, total_b / n_games