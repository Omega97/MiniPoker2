from mini_poker.game import MiniPoker, State, Trajectory


def test_game(p1_card=0, p2_card=1):
    """Print game's decision points, terminal states and rewards"""
    game = MiniPoker(game_power=2, deck_size=5)
    print()
    for t in game.tree:
        print(f'"{t}"')
    print()
    for branch in game.terminals:
        # Unpack the tuple and print both rewards
        state = State(p1_card, p2_card, branch)
        r1, r2 = game.get_reward(state)
        print(f'"{branch}" ({r1:+.0f}, {r2:+.0f})')


def test_trajectory():
    state = State(1, 2, "")
    trajectory = Trajectory(state)
    trajectory.perform_action("R", .5)
    trajectory.perform_action("D", .5)
    trajectory.perform_action("C", .5)

    # for infoset in trajectory.get_infosets_history():
    #     print(infoset)

    for state in trajectory.get_state_history():
        print(state)


if __name__ == '__main__':
    # test_game(p1_card=0, p2_card=1)
    # test_game(p1_card=1, p2_card=0)
    test_trajectory()
