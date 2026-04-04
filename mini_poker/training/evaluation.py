import numpy as np
import random
from typing import TYPE_CHECKING
from typing import List
from mini_poker.utils import print_colored_status
from mini_poker.game import MiniPoker, Infoset, State

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from mini_poker.agents.base_agent import BaseAgent


def _evaluate_agents_fixed_position(game: MiniPoker, agents: List['BaseAgent'], epochs=1):
    """
    Plays two agents against each other for n_games.
    Returns the average reward per game for (Player 1, Player 2).
    """
    for agent in agents:
        assert hasattr(agent, "get_action"), f"agent {agent} does not have 'get_action'"

    total_rewards_p1 = []
    total_rewards_p2 = []

    for p1_card, p2_card in game.iter_uniformly_over_hands(n_sweeps=epochs):
        # 1. Deal random cards (permutations ensures no duplicate cards)
        state = State(p1_card, p2_card, branch="")

        # 2. Play the game until a terminal state is reached
        while not game.is_terminal(state.branch):
            # Determine whose turn it is
            acting_player_idx = state.get_current_player()

            # Select the correct agent and card for the current turn
            current_agent = agents[acting_player_idx]

            # Get action based on the agent's learned policy
            infoset = state.get_current_player_infoset()
            action = current_agent.get_action(infoset)
            state = state.perform_action(action)

        # 3. Calculate rewards from the terminal state
        r1, r2 = game.get_reward(state)
        total_rewards_p1.append(r1)
        total_rewards_p2.append(r2)

    avg_rewards_p1 = np.mean(total_rewards_p1)
    avg_rewards_p2 = np.mean(total_rewards_p2)
    n_games = len(total_rewards_p1)

    return avg_rewards_p1, avg_rewards_p2, n_games


def evaluate_agents(game: MiniPoker, agents: List['BaseAgent'], epochs=1):
    """
    Evaluates two agents by swapping positions halfway through to eliminate
    positional bias. Returns the net average for Agent A and Agent B.
    """

    # Leg 1: Agent A is P1, Agent B is P2
    p1_score_leg1, p2_score_leg1, n_games = _evaluate_agents_fixed_position(game, agents, epochs=epochs)

    # Leg 2: Agent B is P1, Agent A is P2
    p1_score_leg2, p2_score_leg2, n_games = _evaluate_agents_fixed_position(game, list(reversed(agents)), epochs=epochs)

    # Average performance for Agent A
    # (P1 score when A was first + P2 score when A was second) / 2
    avg_a = float(p1_score_leg1 + p2_score_leg2) / 2
    avg_b = float(p2_score_leg1 + p1_score_leg2) / 2

    return avg_a, avg_b, n_games * 2


def all_v_all_tournament(game: MiniPoker, agents: List['BaseAgent'], epochs=1, random_seed=0):
    """Play games between all agents."""
    random.seed(random_seed)
    rewards_sum = np.zeros((len(agents), len(agents)))
    n_games = None
    total_games = 0

    while True:

        # Play games
        for i in range(len(agents)):
            agent_1 = agents[i]
            for j in range(i):
                agent_2 = agents[j]
                avg_p1, avg_p2, n_games = evaluate_agents(game, [agent_1, agent_2], epochs=epochs)
                rewards_sum[i, j] += avg_p1 * n_games
                rewards_sum[j, i] += avg_p2 * n_games
        total_games += n_games

        # Print
        print('\n')
        print(f"Results after {total_games} games\n")
        length = max([len(str(a)) for a in agents])
        for i in range(len(agents)):
            print(f"{str(agents[i]):>{length}} ", end="")
            for j in range(len(agents)):
                score = float(rewards_sum[i, j]) / total_games
                s = print_colored_status(round(score, 2), text=f"{score:+5.2f}")
                print(f" {s}", end=' ')
            print()

        epochs += 1


def quick_evaluate_agents(game: MiniPoker, agents: List['BaseAgent'], n_games=10_000):
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
            agent_p1, agent_p2 = agents
            card_p1, card_p2 = p1_card, p2_card
        else:
            agent_p1, agent_p2 = reversed(agents)
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
