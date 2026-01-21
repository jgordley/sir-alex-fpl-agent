"""Configuration settings for Sir Alex FPL Agent."""

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "Sir Alex - FPL Agent"
    allowed_users: list[str] = Field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Parse ALLOWED_USERS from comma-separated string
        allowed_users_str = os.getenv("ALLOWED_USERS", "")
        allowed_users = [
            user.strip()
            for user in allowed_users_str.split(",")
            if user.strip()
        ]

        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            allowed_users=allowed_users,
        )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings.from_env()


def is_valid_user(user_id: str) -> bool:
    """Check if a user ID is in the allowed users list.

    Args:
        user_id: The user/actor ID to validate.

    Returns:
        True if the user is allowed, False otherwise.
        If ALLOWED_USERS is empty/not set, all users are allowed.
    """
    settings = get_settings()
    # If no allowed users configured, allow everyone
    if not settings.allowed_users:
        return True
    return user_id in settings.allowed_users
