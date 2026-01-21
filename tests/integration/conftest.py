"""Pytest configuration for integration tests."""

import os

import pytest

# Model used for integration tests
TEST_MODEL = "anthropic/claude-haiku-4.5"


@pytest.fixture(scope="session")
def openrouter_api_key():
    """Get OpenRouter API key, skip if not available."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set - skipping integration tests")
    return api_key


@pytest.fixture(scope="session")
def test_model():
    """Return the model name used for integration tests."""
    return TEST_MODEL
