
# 🃏 MiniPoker2: CFR-Based Poker Intelligence

**MiniPoker2** is a framework for training and playing against superhuman AI in a simplified poker environment. Built on the principles of **Counterfactual Regret Minimization (CFR)**, the bot doesn't just play "well"—it plays an unexploitable Nash Equilibrium strategy.

---

## 🚀 Key Features

### 🧠 Strategic Intelligence (CFR)
The core engine is powered by a **CFR (Counterfactual Regret Minimization)** algorithm. Unlike standard Reinforcement Learning, CFR is specifically designed for **Imperfect Information Games**. Through millions of iterations of self-play, the bot learns to balance its range, effectively mixing bluffs and value bets to remain mathematically unexploitable.

### 🔍 Real-Time Hand Posteriors
Ever wonder what the bot is thinking? You can query the bot to see its **Posterior Distribution** over your possible hands. By analyzing the history of the game ($h$) and its own policy, the bot calculates the probability of you holding every card in the deck.
* **Feature:** Use `ask_the_policy` in `main.py` to see the bot’s internal belief state.

### 🕹️ Human vs. AI Interface
Ready to test your skills? The `PlayVsAI` module allows you to go head-to-head with the bot in the console. 
* **Detailed Logging:** Every hand is saved to a `.json` file, including the cards dealt, the sequence of moves, and the resulting EV (Expected Value).
* **Performance Tracking:** The program generates a Matplotlib graph of your cumulative reward to show if you're actually beating the bot over the long run (Spoiler: You probably won't).

### 🃏 IRL Assistant (The "Friend Crusher")
Because the bot handles standard poker notation (e.g., `History: 'CRR'`), you can use it as a real-time advisor while playing against friends in real life. Input the board state and the betting history, and let the CFR engine tell you the mathematically optimal action to take.

---

## 🛠️ How to Use

### 1. Play Against the Bot
Simply run the main script to start an interactive session:
```bash
python main.py
```
* Use **Enter** to progress through AI turns.
* Type **'q'** to quit and view your performance analytics.

### 2. Query the Oracle
To ask the bot for advice or see its hand probabilities for a specific scenario, use the `ask_the_policy` function in `main.py`:
```python
# Example: What should I do with card 33 if the history is 'Check-Raise-Raise'?
ask_the_policy(card=33, history="CRR", agent=ai_agent)
```

### 3. Review History
After playing, the bot stores a detailed summary in your `GAME_DATA_DIR`. You can view the table of past hands (including what cards everyone had) by calling:
```python
PlayVsAI(ai_agent).display_history()
```

---

## 📈 Technical Insight: Why the Bot is Hard to Beat
The bot utilizes **Mixed Strategies**. In certain spots, you might see the bot take an action that seems lower in EV than another. This is a theoretical necessity of the Nash Equilibrium; the bot "sacrifices" immediate greed to ensure its overall range is balanced, making it impossible for you to "read" its hand based on its bets.



---

## 📁 Project Structure
* `mini_poker/agents/`: Contains the CFR, CRM, and EM agent implementations.
* `mini_poker/game.py`: The logic for the Mini-Poker environment and reward calculation.
* `scripts/`: Utility scripts for loading trained models and analyzing policies.
* `main.py`: The central entry point for play and analysis.

---

*“In poker, you don’t play your hand, you play the man across from you. But against CFR, there is no man—only math.”*
