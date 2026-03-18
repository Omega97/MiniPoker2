

def clip(x, low, high):
    return max(low, min(high, x))


def print_colored_status(text: str, value: float):
    """
    Prints text in color based on the sign of value.
    Red for negative, Green for positive, White for zero.
    """
    # ANSI escape code sequences
    RED = "\033[91m"
    GREEN = "\033[92m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    if value < 0:
        color = RED
    elif value > 0:
        color = GREEN
    else:
        color = WHITE

    return f"{color}{text}{RESET}"


def card_to_num(rank: str, suit: str) -> int:
    """
    Converts a card's rank and suit to a unique integer index [0-51].

    Ranks: 2-10, J, Q, K, A
    Suits: s (spades), h (hearts), d (diamonds), c (clubs)
    """
    # Define the ordering for ranks
    rank_map = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
        '7': 5, '8': 6, '9': 7, '10': 8, 'T': 8,
        'J': 9, 'Q': 10, 'K': 11, '1': 12, 'A': 12
    }

    # Define the ordering for suits (Spades, Clubs, Diamonds, Hearts)
    suit_map = {'s': 0, 'c': 1, 'd': 2, 'h': 3}

    # Normalize inputs
    rank = str(rank).upper()
    suit = str(suit).lower()

    if rank not in rank_map or suit not in suit_map:
        raise ValueError(f"Invalid card: {rank}{suit}")

    # Standard formula: (Rank Index * Number of Suits) + Suit Index
    return rank_map[rank] * 4 + suit_map[suit]


def num_to_card(num: int):
    """Reverses the process: converts an index back to (rank, suit)."""
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']

    rank = ranks[num // 4]
    suit = suits[num % 4]
    return rank, suit
