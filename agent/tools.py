"""Tools for Sir Alex FPL Agent."""

from langchain_core.tools import tool


@tool
def math_add(x: float, y: float) -> float:
    """Add two numbers together.

    Args:
        x: The first number to add.
        y: The second number to add.

    Returns:
        The sum of x and y.
    """
    return x + y


# List of all available tools
ALL_TOOLS = [math_add]
