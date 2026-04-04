from mini_poker.ask_policy import ask_the_policy
from mini_poker.utils import card_to_num
from scripts.load_good_agent import load_good_agent


def get_input_normalized(prompt):
    """Clean up user input for cards and branches."""
    return input(prompt).strip()


def run_interactive_oracle():
    print("=== MINI-POKER AI ORACLE ===")
    print("Loading agent (Game Power: 5, Deck: 52)...")

    try:
        agent = load_good_agent(game_power=5, deck_size=52)
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    print("\nInstructions:")
    print("- Enter hand as a number (0-51) or string (e.g., 'Ah', '10s').")
    print("- Enter branch as a sequence of actions (e.g., 'CR', 'C').")
    print("- For root state, use '', '-', or 'root'.")
    print("- To quit: Press Enter twice when asked for a branch.")

    last_was_empty = False

    while True:
        # 1. Get Hand
        hand_input = get_input_normalized("\nEnter your hand: ")
        if not hand_input:
            print("Hand cannot be empty.")
            continue

        # 2. Get Branch
        branch_input = get_input_normalized(f"Enter branch for hand {hand_input}: ")

        # Check for exit condition (Double empty branch)
        if branch_input == "":
            if last_was_empty:
                print("Exiting Oracle. Good luck at the table!")
                break
            last_was_empty = True
        else:
            last_was_empty = False

        # Normalize root state aliases
        if branch_input.lower() in ["", "-", "root"]:
            branch_processed = ""
        else:
            branch_processed = branch_input.upper()

        # 3. Validation and Execution
        try:
            # Handle string vs int conversion for hand
            if hand_input.isdigit():
                final_hand = int(hand_input)
            else:
                # Assuming card_to_num handles ranks like 'Ah' -> 'A', 'h'
                # and you might need to slice the string if your utils expect separate args
                if len(hand_input) == 2:
                    final_hand = card_to_num(hand_input[0], hand_input[1])
                elif len(hand_input) == 3:  # For '10s'
                    final_hand = card_to_num(hand_input[:2], hand_input[2])
                else:
                    final_hand = hand_input

            print(f"\n--- Analysis for {hand_input} at '{branch_processed or 'root'}' ---")

            # Call the analysis function
            ask_the_policy(
                my_hand=final_hand,
                branch=branch_processed,
                agent=agent
            )

        except KeyError:
            print(f"Error: Branch '{branch_input}' is not a valid node in the game tree.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_interactive_oracle()
