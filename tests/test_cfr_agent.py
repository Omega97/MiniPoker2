import time
import numpy as np
from mini_poker.game import MiniPoker, Infoset, ACTIONS
from mini_poker.training.evaluation import random_games_evaluate_agents
from mini_poker.utils import print_colored_status, num_to_card
from mini_poker.agents.crm_agent import CRMAgent
from mini_poker.agents.cfr_agent import CFRAgent


def test_1_eval_search(n_eval_games=20_000, jump_factor=1.1):
    game = MiniPoker(5, 52)

    # Load same checkpoint for both agents

    # agent_1 = CFRAgent(game, epochs=1000, n_moves_no_search=2)  # +0.18
    # agent_1.load()
    # agent_1.set_search_enabled(True, iterations=1000)

    # agent_1 = CFRAgent(game, epochs=1000, n_moves_no_search=2)  # +0.04
    # agent_1.load()
    # agent_1.set_search_enabled(True, iterations=100)

    agent_1 = CFRAgent(game, epochs=1000, n_moves_no_search=1)  # + 0.30
    agent_1.load()
    agent_1.set_search_enabled(True, iterations=1000)

    agent_2 = CFRAgent(game, epochs=1000)  # +0.00
    agent_2.load()

    print(f"Agent 1: {agent_1.name}")
    print(f"Agent 2: {agent_2.name}")
    print("-" * 60)

    start_time = time.time()

    period = 1
    results = []

    # Start with 10,000 games for quick testing, scale to 100k if signal is clear
    for r1, r2, n in random_games_evaluate_agents(game, [agent_1, agent_2], n_games=n_eval_games):
        if n >= period or n == n_eval_games-1:
            elapsed = time.time() - start_time
            if elapsed > 0:
                games_per_sec = n / elapsed
                str_r1 = print_colored_status(r1, f"{r1:+.2f}")
                print(f"{n+1:>6}) {str_r1} {r2:+.2f}  ({games_per_sec:.1f} games/s)")
            results.append((r1, r2, n))
            period = int(period * jump_factor) + 1

    final_r1, final_r2, _ = results[-1]
    elapsed = time.time() - start_time

    print("-" * 60)
    print(f"FINAL: Search EV = {final_r1:+.3f}, Baseline EV = {final_r2:+.3f}")
    print(f"Total Time: {elapsed / 60:.2f} min")

    return results


def test_2_policy_change_with_search(card=48, branch="RRRR", n_iterations=100):
    """
    Show the difference in policy before and after online CFR search
    for an obscure infoset that likely has sparse training data.

    Pick an obscure infoset: high card + aggressive history
    """
    infoset = Infoset(card, branch)
    game = MiniPoker(5, 52)

    # Load trained agent
    agent = CFRAgent(game, epochs=1000, search_enabled=False)
    agent.load()

    print("=" * 70)
    print(f"POLICY CHANGE ANALYSIS: Infoset{infoset}")
    print("=" * 70)
    print(f"Card: {card} / {game.deck_size - 1} (top {100 * (card / game.deck_size):.0f}%)")
    print(f"History: '{branch}' ({len(branch)} actions - rare situation)")
    print()

    # Check if this infoset exists in training
    if infoset not in agent.cumulative_policy:
        print(f"⚠️  Infoset{infoset} was NEVER visited during training!")
        print(f"   Search will be working with zero prior data.")
    else:
        visits = sum(agent.reward_counts[infoset].values())
        print(f"✓  Infoset visited {visits} times during training")
        if visits < 100:
            print(f"   ⚠️  Sparse data - search should help significantly")
    print()

    # 1. Get policy BEFORE search (average policy from training)
    policy_before = agent.get_average_policy(infoset)

    print("📋 POLICY BEFORE SEARCH (Average Policy from Training):")
    print("-" * 70)
    print(f"{'Action':<10} {'Probability':<15} {'Cumulative Policy':<20} {'Regret':<15}")
    print("-" * 70)
    for action in sorted(policy_before.keys()):
        prob = policy_before[action]
        cum_policy = agent.cumulative_policy[infoset].get(action, 0.0)
        regret = agent.cumulative_regrets[infoset].get(action, 0.0)
        prob_str = f"{prob:>6.1%}"
        print(f"{action:<10} {prob_str:<15} {cum_policy:<20.2f} {regret:<+15.2f}")
    print()

    # 2. Compute posterior over opponent cards
    posterior = agent.compute_posterior(infoset)
    print("🃏 POSTERIOR OVER OPPONENT CARDS (Top 5):")
    print("-" * 70)
    top_cards = np.argsort(posterior)[-5:][::-1]
    for card in top_cards:
        if posterior[card] > 0.01:
            bar = "█" * int(posterior[card] * 50)
            print(f"  Card {card:2d}: {posterior[card]:5.1%} {bar}")
    print()

    # 3. Enable search and get policy AFTER search
    agent.set_search_enabled(True, iterations=n_iterations)
    policy_after = agent.online_search(infoset, n_iterations=n_iterations)
    agent.set_search_enabled(False)  # Disable again

    print("🔍 POLICY AFTER SEARCH (500 Online CFR Iterations):")
    print("-" * 70)
    print(f"{'Action':<10} {'Probability':<15} {'Change':<15} {'Direction':<10}")
    print("-" * 70)
    for action in sorted(policy_after.keys()):
        prob_before = policy_before[action]
        prob_after = policy_after[action]
        change = prob_after - prob_before

        if change > 0.05:
            change_str = f"\033[92m{change:+.1%}\033[0m"  # Green increase
            direction = "↑ INCREASE"
        elif change < -0.05:
            change_str = f"\033[91m{change:+.1%}\033[0m"  # Red decrease
            direction = "↓ DECREASE"
        else:
            change_str = f"\033[93m{change:+.1%}\033[0m"  # Yellow small change
            direction = "→ STABLE"

        prob_str = f"{prob_after:>6.1%}"

        print(f"{action:<10} {prob_str:<15} {change_str:<15} {direction:<10}")
    print()

    # 4. Summary statistics
    print("📊 SUMMARY STATISTICS:")
    print("-" * 70)
    total_change = sum(abs(policy_after[a] - policy_before[a]) for a in policy_before)
    max_change_action = max(policy_before.keys(), key=lambda a: abs(policy_after[a] - policy_before[a]))
    max_change = abs(policy_after[max_change_action] - policy_before[max_change_action])

    print(f"  Total Policy Change (L1 norm):     {total_change:.3f}")
    print(f"  Max Change (action '{max_change_action}'):       {max_change:.1%}")
    print()

    # 5. Interpretation
    print("💡 INTERPRETATION:")
    print("-" * 70)
    if total_change < 0.1:
        print("  Search made MINIMAL changes - policy was already well-trained")
        print("  for this infoset, or search iterations were too few.")
    elif total_change < 0.5:
        print("  Search made MODERATE changes - refining edge cases in the policy.")
    else:
        print("  Search made MAJOR changes - this infoset was poorly trained")
        print("  and search is significantly correcting the strategy.")

    if infoset not in agent.cumulative_policy or sum(agent.reward_counts[infoset].values()) < 50:
        print("  ⚠️  This infoset had sparse training data - search is compensating.")
    print()
    print("=" * 70)
    print('\n')

    return policy_before, policy_after, total_change


def test_3_policy_change(card, branch, n_iterations=500):
    """
    Display action frequencies before and after online CFR search,
    then sample and print a random action based on the refined policy.

    :param card: Your card (0-51)
    :param branch: Action history (e.g., "RRR")
    :param n_iterations: Search iterations
    """
    game = MiniPoker(5, 52)
    agent = CFRAgent(game, epochs=2000)
    agent.load()

    infoset = Infoset(card, branch)

    # Policy BEFORE search
    policy_before = agent.get_average_policy(infoset)

    # Policy AFTER search
    # Ensure search methods exist in your CRMAgent implementation
    if hasattr(agent, 'set_search_enabled'):
        agent.set_search_enabled(True, iterations=n_iterations)
        policy_after = agent.online_search(infoset, n_iterations=n_iterations)
        agent.set_search_enabled(False)
    else:
        # Fallback if search isn't implemented in this specific file version yet
        print("⚠️ Search methods not found. Showing trained policy only.")
        policy_after = policy_before

    # Display Table
    visits = sum(agent.reward_counts[infoset].values()) if infoset in agent.reward_counts else 0
    print(f"\nInfoset: Card={num_to_card(card)} ({card}), History='{branch}'")
    print(f"Training visits: {visits}")
    print(f"\n{'Action':<8} {'Before':<10} {'After':<10}")
    print("-" * 30)

    for action in ACTIONS:
        if action in policy_before:
            before = policy_before[action]
            after = policy_after[action]
            print(f"{action:<8} {before:>8.1%}  {after:>8.1%}")

    total_change = sum(abs(policy_after[a] - policy_before[a]) for a in policy_before)
    print(f"\nTotal change (L1): {total_change:.3f}")

    # Sample Random Action from the Refined Policy
    actions_list = list(policy_after.keys())
    probs_list = list(policy_after.values())
    recommended_action = np.random.choice(actions_list, p=probs_list)

    print(f'\n({num_to_card(card)}, "{branch}")  →  RANDOM ACTION:  "{recommended_action}"')


if __name__ == '__main__':
    test_1_eval_search()
    # test_2_policy_change_with_search(card=40, branch="RRR", n_iterations=200)
    # test_3_policy_change(card=40, branch="RRR", n_iterations=1)
