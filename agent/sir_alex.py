"""Sir Alex LangGraph Agent."""

import os
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_checkpoint_aws import AgentCoreMemorySaver
from pydantic import BaseModel, Field

from agent.tools import ALL_TOOLS


def get_checkpointer():
    """Get the AgentCore Memory checkpointer if configured."""
    memory_id = os.getenv("AGENTCORE_MEMORY_ID")
    if not memory_id:
        return None
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    return AgentCoreMemorySaver(memory_id, region_name=region)


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


def create_agent(model_name: str = "anthropic/claude-haiku-4.5", checkpointer=None):
    """Create the Sir Alex agent graph.

    Args:
        model_name: The model to use via OpenRouter.
        checkpointer: Optional checkpointer for state persistence.

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

    return graph.compile(checkpointer=checkpointer)


def get_conversation_history(
    model_name: str = "anthropic/claude-haiku-4.5",
    actor_id: str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Load conversation history from AgentCore memory.

    Args:
        model_name: The model to use via OpenRouter.
        actor_id: User/actor identifier for memory persistence.
        session_id: Session/thread identifier for conversation continuity.

    Returns:
        List of message dicts with 'role' and 'content' keys.
    """
    checkpointer = get_checkpointer()
    if not checkpointer or not actor_id or not session_id:
        return []

    agent = create_agent(model_name, checkpointer=checkpointer)
    config = {
        "configurable": {
            "thread_id": session_id,
            "actor_id": actor_id,
        }
    }

    try:
        state = agent.get_state(config)
        if not state or not state.values:
            return []

        messages = state.values.get("messages", [])
        history = []
        pending_tool_calls = {}  # tool_call_id -> {name, args}

        for msg in messages:
            if not hasattr(msg, "type"):
                continue

            if msg.type == "human":
                history.append({"role": "user", "content": msg.content})

            elif msg.type == "ai":
                # Track tool calls for later matching with results
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        pending_tool_calls[tc_id] = {
                            "name": tc["name"],
                            "args": tc["args"],
                            "result": "",
                        }

                # Only add AI messages with actual content
                if msg.content:
                    assistant_msg = {"role": "assistant", "content": msg.content}
                    # Attach any completed tool calls
                    completed_tools = [
                        tc for tc in pending_tool_calls.values() if tc["result"]
                    ]
                    if completed_tools:
                        assistant_msg["tool_calls"] = completed_tools
                        pending_tool_calls = {}  # Reset for next turn
                    history.append(assistant_msg)

            elif msg.type == "tool":
                # Match tool result to pending tool call
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in pending_tool_calls:
                    pending_tool_calls[tc_id]["result"] = str(msg.content)

        return history
    except Exception:
        return []


def run_agent(
    user_message: str,
    model_name: str = "anthropic/claude-haiku-4.5",
    actor_id: str | None = None,
    session_id: str | None = None,
) -> AgentResponse:
    """Run the agent with a user message.

    Args:
        user_message: The user's input message.
        model_name: The model to use via OpenRouter.
        actor_id: User/actor identifier for memory persistence.
        session_id: Session/thread identifier for conversation continuity.

    Returns:
        AgentResponse with content and tool calls.
    """
    checkpointer = get_checkpointer()
    agent = create_agent(model_name, checkpointer=checkpointer)

    # Build config for memory if actor_id and session_id are provided
    config = None
    existing_message_count = 0
    if checkpointer and actor_id and session_id:
        config = {
            "configurable": {
                "thread_id": session_id,
                "actor_id": actor_id,
            }
        }
        # Get existing message count before invoke
        try:
            state = agent.get_state(config)
            if state and state.values:
                existing_message_count = len(state.values.get("messages", []))
        except Exception:
            pass

    result = agent.invoke({"messages": [("user", user_message)]}, config=config)

    # Extract tool calls and their results from NEW messages only
    tool_calls = []
    messages = result["messages"]
    new_messages = messages[existing_message_count:]  # Only process new messages

    for i, message in enumerate(new_messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Find the corresponding tool result
                tool_result = ""
                for next_msg in new_messages[i + 1 :]:
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

    # Get final response from new messages
    final_content = "I couldn't generate a response."
    for message in reversed(new_messages):
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
