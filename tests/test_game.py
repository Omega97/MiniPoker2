from mini_poker.game import MiniPoker


def test_game(p1_card=0, p2_card=1):
    """Print game's decision points, terminal states and rewards"""
    game = MiniPoker(game_power=2, deck_size=5)
    print()
    for t in game.tree:
        print(f'"{t}"')
    print()
    for history in game.terminals:
        # Unpack the tuple and print both rewards
        r1, r2 = game.get_reward(history, p1_card, p2_card)
        print(f'"{history}" ({r1:+.0f}, {r2:+.0f})')


if __name__ == '__main__':
    test_game(p1_card=0, p2_card=1)
    test_game(p1_card=1, p2_card=0)
