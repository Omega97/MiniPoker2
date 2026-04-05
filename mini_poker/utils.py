import numpy as np


# ANSI escape code sequences
COLORS = {
    # Foreground Colors (Standard)
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "black": "\033[30m",

    # Foreground Colors (Bright/Bold)
    "bright_red": "\033[91;1m",
    "bright_green": "\033[92;1m",
    "bright_yellow": "\033[93;1m",
    "bright_blue": "\033[94;1m",
    "bright_magenta": "\033[95;1m",
    "bright_cyan": "\033[96;1m",
    "bright_white": "\033[97;1m",

    # Foreground Colors (Dark/Dim)
    "dark_red": "\033[31m",
    "dark_green": "\033[32m",
    "dark_yellow": "\033[33m",
    "dark_blue": "\033[34m",
    "dark_magenta": "\033[35m",
    "dark_cyan": "\033[36m",
    "dark_white": "\033[37m",

    # Background Colors
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
    "bg_white": "\033[47m",
    "bg_black": "\033[40m",

    # Background Colors (Bright)
    "bg_bright_red": "\033[101m",
    "bg_bright_green": "\033[102m",
    "bg_bright_yellow": "\033[103m",
    "bg_bright_blue": "\033[104m",
    "bg_bright_magenta": "\033[105m",
    "bg_bright_cyan": "\033[106m",
    "bg_bright_white": "\033[107m",

    # Text Styles
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "blink": "\033[5m",
    "reverse": "\033[7m",
    "hidden": "\033[8m",
    "strikethrough": "\033[9m",

    # Reset Codes
    "reset": "\033[0m",
    "reset_color": "\033[39m",
    "reset_bg": "\033[49m",
    "reset_style": "\033[22m",
}

# Convenience aliases
COLORS["orange"] = COLORS["yellow"]  # Terminal doesn't have true orange
COLORS["purple"] = COLORS["magenta"]
COLORS["grey"] = COLORS["dark_white"]
COLORS["gray"] = COLORS["dark_white"]
COLORS["normal"] = COLORS["reset"]


def clip(x, low, high):
    return max(low, min(high, x))


def print_colored_status(value: float, text: str = None,
                         green=COLORS["green"], red=COLORS["red"]):
    """
    Prints text in color based on the sign of value.
    Red for negative, Green for positive, White for zero.
    """
    if value < 0:
        color = red
    elif value > 0:
        color = green
    else:
        color = None

    if text is None:
        text = str(value)

    if color is None:
        return text
    else:
        return f"{color}{text}{COLORS['reset']}"


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
    suits = ['♠️', '♣️', '♦️', '♥️']

    rank = ranks[num // 4]
    suit = suits[num % 4]
    return rank + suit


def clip_proba(a, threshold: float = 1e-3) -> np.ndarray:
    """Clip probabilities below threshold to zero and renormalize."""
    arr = np.asarray(a, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("Input must be 1D")

    arr[arr <= threshold] = 0.0
    total = arr.sum()

    if total == 0:
        raise ValueError("Cannot normalize: total probability is zero after clipping")

    return arr / total
