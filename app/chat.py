"""Chat functionality using OpenRouter API."""

from openai import OpenAI

from app.config import get_settings
from app.constants import SYSTEM_PROMPT


def get_openrouter_client() -> OpenAI:
    """Create an OpenRouter client using OpenAI SDK."""
    settings = get_settings()
    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )


def get_chat_response(
    messages: list[dict],
    model: str,
    user_id: str | None = None,
    fpl_team_id: str | None = None,
) -> str:
    """
    Get a chat response from OpenRouter.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: The model identifier (e.g., 'anthropic/claude-haiku-4.5')
        user_id: Optional user identifier for tracking
        fpl_team_id: Optional FPL team ID for context

    Returns:
        The assistant's response text
    """
    client = get_openrouter_client()

    # Build system message with optional FPL context
    system_content = SYSTEM_PROMPT
    if fpl_team_id:
        system_content += f"\n\nThe user's FPL Team ID is: {fpl_team_id}"

    # Prepend system message
    full_messages = [{"role": "system", "content": system_content}] + messages

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        extra_headers={
            "HTTP-Referer": "https://sir-alex-fpl.app",
            "X-Title": "Sir Alex FPL Agent",
        },
    )

    return response.choices[0].message.content
