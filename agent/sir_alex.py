"""Sir Alex LangGraph Agent."""

import os
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from agent.tools import ALL_TOOLS


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""

    name: str
    args: dict
    result: str


class AgentResponse(BaseModel):
    """Response from the agent including tool calls."""

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AgentState(TypedDict):
    """State for the Sir Alex agent."""

    messages: Annotated[list, add_messages]


def create_agent(model_name: str = "anthropic/claude-haiku-4.5"):
    """Create the Sir Alex agent graph.

    Args:
        model_name: The model to use via OpenRouter.

    Returns:
        Compiled LangGraph agent.
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

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(
    user_message: str, model_name: str = "anthropic/claude-haiku-4.5"
) -> AgentResponse:
    """Run the agent with a user message.

    Args:
        user_message: The user's input message.
        model_name: The model to use via OpenRouter.

    Returns:
        AgentResponse with content and tool calls.
    """
    agent = create_agent(model_name)

    result = agent.invoke({"messages": [("user", user_message)]})

    # Extract tool calls and their results
    tool_calls = []
    messages = result["messages"]

    for i, message in enumerate(messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Find the corresponding tool result
                tool_result = ""
                for next_msg in messages[i + 1 :]:
                    if hasattr(next_msg, "name") and next_msg.name == tc["name"]:
                        tool_result = str(next_msg.content)
                        break

                tool_calls.append(
                    ToolCall(
                        name=tc["name"],
                        args=tc["args"],
                        result=tool_result,
                    )
                )

    # Get final response
    final_content = "I couldn't generate a response."
    for message in reversed(messages):
        if hasattr(message, "content") and message.content:
            if not hasattr(message, "tool_calls") or not message.tool_calls:
                final_content = message.content
                break

    return AgentResponse(content=final_content, tool_calls=tool_calls)


if __name__ == "__main__":
    response = run_agent("What is 5 + 3?")
    print(f"Response: {response.content}")
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}({tc.args}) -> {tc.result}")
