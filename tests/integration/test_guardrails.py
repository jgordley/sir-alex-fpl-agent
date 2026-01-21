"""Integration tests for agent guardrails."""

import pytest

from agent import run_agent


@pytest.mark.integration
class TestGuardrails:
    """Test that the guardrails correctly filter on-topic and off-topic queries."""

    def test_on_topic_query_passes_guardrail(self, openrouter_api_key, test_model):
        """Test that a soccer-related query passes the guardrail.

        When user asks about Haaland's performance, the guardrail should classify
        it as ON_SOCCER_TOPIC and allow the agent to process it.
        """
        response = run_agent(
            user_message="How is Haaland performing this season?",
            model_name=test_model,
        )

        # Should NOT contain the guardrail rejection message
        assert (
            "I'm Sir Alex Ferguson, and I'm here to help you with Fantasy Premier League"
            not in response.content
        )
        # Should contain some actual response about Haaland or stats
        assert response.content is not None
        assert len(response.content) > 50  # Should be a substantive response

    def test_off_topic_query_blocked_by_guardrail(self, openrouter_api_key, test_model):
        """Test that an off-topic query is blocked by the guardrail.

        When user asks about cooking spaghetti, the guardrail should classify
        it as DO_NOT_ANSWER and return the rejection message.
        """
        response = run_agent(
            user_message="How do I cook spaghetti?",
            model_name=test_model,
        )

        # Should contain the guardrail rejection message
        assert "I'm Sir Alex Ferguson" in response.content
        assert (
            "Fantasy Premier League" in response.content
            or "football" in response.content.lower()
            or "soccer" in response.content.lower()
        )
        # Should have no tool calls since we didn't invoke the agent
        assert len(response.tool_calls) == 0
