"""Integration tests for agent tool selection accuracy using AgentEvals."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator

from agent import create_agent


@pytest.mark.integration
class TestPlayerSearchToolSelection:
    """Test that the agent correctly selects tools and arguments for player searches."""

    @pytest.fixture
    def agent(self, openrouter_api_key, test_model):
        """Create the agent for testing."""
        return create_agent(model_name=test_model)

    @pytest.fixture
    def trajectory_evaluator(self):
        """Create a trajectory match evaluator in unordered mode."""
        return create_trajectory_match_evaluator(
            trajectory_match_mode="unordered",
            tool_args_match_mode="exact",
        )

    def test_historical_season_query_includes_season_arg(self, agent, trajectory_evaluator):
        """Test that asking about a specific year includes the season argument.

        When user asks "How did Haaland do in 2024", the agent should call
        get_player_stats with season="24/25".
        """
        result = agent.invoke({
            "messages": [HumanMessage(content="How did Haaland do in 2024?")]
        })

        # Define expected trajectory - we expect the tool to be called with season arg
        reference_trajectory = [
            HumanMessage(content="How did Haaland do in 2024?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_player_stats",
                        "args": {"player_name": "Haaland", "season": "24/25"},
                    }
                ],
            ),
            ToolMessage(content="", tool_call_id="call_1"),
            AIMessage(content=""),
        ]

        evaluation = trajectory_evaluator(
            outputs=result["messages"],
            reference_outputs=reference_trajectory,
        )

        assert evaluation["score"] is True, (
            f"Expected agent to call get_player_stats with season='24/25'. "
            f"Actual trajectory: {result['messages']}"
        )

    def test_current_performance_query_no_season_arg(self, agent, trajectory_evaluator):
        """Test that asking about current performance does not include season argument.

        When user asks "How is Haaland's current performance?", the agent should call
        get_player_stats without a season argument.
        """
        result = agent.invoke({
            "messages": [HumanMessage(content="How is Haaland's current performance?")]
        })

        # Define expected trajectory - we expect the tool to be called WITHOUT season arg
        reference_trajectory = [
            HumanMessage(content="How is Haaland's current performance?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_player_stats",
                        "args": {"player_name": "Haaland"},
                    }
                ],
            ),
            ToolMessage(content="", tool_call_id="call_1"),
            AIMessage(content=""),
        ]

        evaluation = trajectory_evaluator(
            outputs=result["messages"],
            reference_outputs=reference_trajectory,
        )

        assert evaluation["score"] is True, (
            f"Expected agent to call get_player_stats without season arg. "
            f"Actual trajectory: {result['messages']}"
        )
