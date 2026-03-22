import random
from mini_poker.agents.base_agent import BaseAgent, Infoset
from mini_poker.game import State


class HumanAgent(BaseAgent):
    """
    An agent that allows a human player to choose actions via the console.
    """

    def __init__(self, game):
        # Human agents don't need logits, but we call super for compatibility
        super().__init__(game)

    def get_action(self, infoset: Infoset):
        """Overrides the sampling logic with manual input."""
        card, history = infoset.get_values()
        legal_actions = self.game.tree[history]

        print(f"\n--- YOUR TURN ---")
        print(f"✋ Hand: [ {card} ] | History: '{history or '(root)'}'")
        print(f"Legal Actions: {', '.join(legal_actions)}")

        action = ""
        while action not in legal_actions:
            action = input(f"Choose action: ").strip().upper()
            if action not in legal_actions:
                print(f"Invalid choice. Please pick from {legal_actions}")

        return action


def play_vs_agent(game, agent_p1, agent_p2):
    """
    Simulates a single game between two agents with console output.
    Can be used to play against a trained CounterfactualAgent.
    """
    # 1. Setup Game State
    cards = random.sample(range(game.deck_size), 2)
    p1_card, p2_card = cards[0], cards[1]
    history = ""

    print("\n" + "=" * 30)
    print(f"GAME START: {agent_p1} vs {agent_p2}")

    # 2. Main Game Loop
    while history not in game.terminals:
        acting_player_idx = len(history) % 2
        current_agent = agent_p1 if acting_player_idx == 0 else agent_p2
        current_card = p1_card if acting_player_idx == 0 else p2_card

        # Identify state
        infoset = Infoset(current_card, history)

        # Logic for non-human agents to show "thinking"
        if not isinstance(current_agent, HumanAgent):
            print(f"\n{current_agent} is thinking...")

        action = current_agent.get_action(infoset)
        history += action
        print(f"Player {acting_player_idx + 1} chose: {action}")
        if not isinstance(current_agent, HumanAgent):
            input()

    # 3. Final Results
    state = State(p1_card, p2_card, branch=history)
    r1, r2 = game.get_reward(state)

    print(f"\nTERMINAL STATE: '{history}'")
    print(f"👀 P1 Card: [{p1_card:2}] | P2 Card: [{p2_card:2}]")
    print(f"REWARDS -> P1: {r1:+.1f}, P2: {r2:+.1f}")
    print("=" * 30)

    return r1, r2
