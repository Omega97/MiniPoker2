

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
