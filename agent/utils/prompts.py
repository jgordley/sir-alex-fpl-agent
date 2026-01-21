"""Prompt utilities for Sir Alex agent."""

from datetime import datetime

from app.constants import SYSTEM_PROMPT
from app.fpl_service import sync_get_current_gameweek


def build_system_prompt(fpl_team_id: str | None = None) -> str:
    """Build the dynamic system prompt with current context.

    Args:
        fpl_team_id: Optional FPL team ID for personalized advice.

    Returns:
        Complete system prompt with date, gameweek, and user context.
    """
    # Get current date/time
    now = datetime.now()
    date_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

    # Get current gameweek
    try:
        gw_info = sync_get_current_gameweek()
        gameweek = gw_info.get("id", "Unknown")
        gw_name = gw_info.get("name", f"Gameweek {gameweek}")
        gw_deadline = gw_info.get("deadline_time", "")
        gameweek_str = f"Current Gameweek: {gw_name}"
        if gw_deadline:
            gameweek_str += f" (Deadline: {gw_deadline})"
    except Exception:
        gameweek_str = "Current Gameweek: Unknown"

    # Build context section
    context_parts = [
        f"Current Date/Time: {date_str}",
        gameweek_str,
    ]

    if fpl_team_id:
        context_parts.append(f"User's FPL Team ID: {fpl_team_id}")

    context = "\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

--- Current Context ---
{context}"""
