"""Node functions for Sir Alex agent graph."""

import os

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from agent.utils.state import AgentState
from agent.utils.tools import ALL_TOOLS


def create_agent_node(model_name: str = "anthropic/claude-haiku-4.5"):
    """Create the agent node function with the specified model.

    Args:
        model_name: The model to use via OpenRouter.

    Returns:
        A node function that processes messages and decides on actions.
    """
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        """Process messages and decide on actions."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return agent_node


def create_tool_node() -> ToolNode:
    """Create the tool node for executing tools.

    Returns:
        A ToolNode configured with all available tools.
    """
    return ToolNode(ALL_TOOLS)
