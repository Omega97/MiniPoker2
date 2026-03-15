import random
from mini_poker.agents.base_agent import Infoset
from mini_poker.utils import print_colored_status


def _evaluate_agents_fixed_position(game, agent_p1, agent_p2, n_games=1000):
    """
    Plays two agents against each other for n_games.
    Returns the average reward per game for (Player 1, Player 2).
    """
    total_reward_p1 = 0
    total_reward_p2 = 0

    # Iterate through games
    for _ in range(n_games):
        # 1. Deal random cards (permutations ensures no duplicate cards)
        # Using game's deck size [cite: 63]
        cards = random.sample(range(game.deck_size), 2)
        p1_card, p2_card = cards[0], cards[1]

        history = ""

        # 2. Play the game until a terminal state is reached [cite: 13]
        while history not in game.terminals:
            # Determine whose turn it is [cite: 10]
            acting_player_idx = len(history) % 2

            # Select the correct agent and card for the current turn
            current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
            current_card = p1_card if acting_player_idx == 0 else p2_card

            # Get action based on the agent's learned policy [cite: 8]
            infoset = Infoset(current_card, history)
            action = current_agent.get_action(infoset)
            history += action

        # 3. Calculate rewards from the terminal state
        r1, r2 = game.get_reward(history, p1_card, p2_card)
        total_reward_p1 += r1
        total_reward_p2 += r2

    return total_reward_p1 / n_games, total_reward_p2 / n_games


def evaluate_agents(game, agent_a, agent_b, n_games=10000):
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


def all_v_all_tournament(game, agents: list, n_games=100_000):
    print(f"\nResults after {n_games} games")
    for i in range(len(agents)):
        agent_1 = agents[i]
        print(f"{str(agent_1):>50}", end=" ")
        for j in range(i):
            agent_2 = agents[j]
            avg_p1, avg_p2 = evaluate_agents(game, agent_1, agent_2, n_games=n_games)
            s = print_colored_status(f"{avg_p1:+5.2f}", round(avg_p1, 2))
            print(f" {s}", end=' ')
        print()
