"""Configuration settings for Sir Alex FPL Agent."""

import os

from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "Sir Alex - FPL Agent"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        return cls(
            openrouter_api_key=api_key,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings.from_env()
