import numpy as np
from mini_poker.agents.batch_kernel_smoothed_methodical_agent import BatchKernelSmoothedMethodicalAgent
from mini_poker.agents.base_agent import Infoset


class PosteriorSamplingAgent(BatchKernelSmoothedMethodicalAgent):
    """
    An agent that performs rollouts by sampling the opponent's
    hand from the current posterior distribution rather than
    assuming a uniform prior.
    """
    def _get_posterior(self, my_card, history):
        """
        Re-uses your Bayesian inference logic to determine
        what the opponent is likely holding.
        """
        game = self.game
        num_cards = game.deck_size
        opponent_probs = np.zeros(num_cards)

        for card_opp in range(num_cards):
            if card_opp == my_card:
                continue

            reach_prob = 1.0
            temp_hist = ""
            for i, action in enumerate(history):
                acting_player = i % 2
                my_player_index = len(history) % 2
                # If it was the opponent's turn in the past
                if acting_player != my_player_index:
                    prev_infoset = Infoset(card_opp, temp_hist)
                    # We use the current policy to see how likely they were to do this
                    probs = self.get_policy(prev_infoset)
                    reach_prob *= probs.get(action, 0.0)
                temp_hist += action
            opponent_probs[card_opp] = reach_prob

        total_weight = np.sum(opponent_probs)
        if total_weight > 0:
            return opponent_probs / total_weight
        else:
            # Fallback to uniform if history is 'impossible'
            post = np.ones(num_cards) / (num_cards - 1)
            post[my_card] = 0
            return post

    def evaluate_action(self, history, action, card1, card2, rollout_samples) -> tuple:
        """
        Overrides the standard evaluation. For each sample, it re-samples
        the opponent's card based on what their actions have revealed so far.
        """
        total_p1 = 0
        total_p2 = 0
        player_turn = len(history) % 2

        # Calculate posterior for the CURRENT player (the one we are evaluating)
        # and for the OPPONENT.
        post_p1 = self._get_posterior(card2, history)
        post_p2 = self._get_posterior(card1, history)

        for _ in range(rollout_samples):
            # Sample "realistic" hands for this rollout
            # If it's P1's turn to be evaluated, we keep card1 but sample a realistic card2
            if player_turn == 0:
                s_card1 = card1
                s_card2 = np.random.choice(range(self.deck_size), p=post_p1)
            else:
                s_card1 = np.random.choice(range(self.deck_size), p=post_p2)
                s_card2 = card2

            temp_h = history + action
            r1, r2 = self.rollout(temp_h, s_card1, s_card2)
            total_p1 += r1
            total_p2 += r2

        return total_p1 / rollout_samples, total_p2 / rollout_samples
