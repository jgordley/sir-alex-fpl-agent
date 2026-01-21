"""Sir Alex LangGraph Agent."""

import logging
import os
import uuid
from datetime import datetime
from typing import TypedDict, Annotated

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore
from pydantic import BaseModel, Field

from agent.tools import ALL_TOOLS
from app.constants import SYSTEM_PROMPT
from app.fpl_service import sync_get_current_gameweek


# Preference strategy ID for long-term memory retrieval
PREFERENCE_STRATEGY_ID = os.getenv("AGENTCORE_PREFERENCE_STRATEGY_ID")


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


def get_checkpointer():
    """Get the AgentCore Memory checkpointer if configured."""
    memory_id = os.getenv("AGENTCORE_MEMORY_ID")
    if not memory_id:
        return None
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    return AgentCoreMemorySaver(memory_id, region_name=region)


def get_memory_store():
    """Get the AgentCore Memory store for long-term memory."""
    memory_id = os.getenv("AGENTCORE_MEMORY_ID")
    if not memory_id:
        return None
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    return AgentCoreMemoryStore(memory_id=memory_id, region_name=region)


def generate_preference_query(user_message: str) -> str:
    """Use LLM to generate a semantic search query for user preferences.

    Args:
        user_message: The user's input message.

    Returns:
        A semantic search query optimized for preference retrieval.
    """
    model_id = os.getenv("SEMANTIC_QUERY_MODEL_ID", "anthropic/claude-haiku-3.5")
    llm = ChatOpenAI(
        model=model_id,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    logger.info(f"User query: {user_message}")

    query_prompt = f"""Based on the following user message about Fantasy Premier League (FPL),
generate a concise semantic search query to find relevant user preferences and past context.
Focus on extracting key topics like: favorite players, teams they support, transfer strategies,
captain preferences, or any FPL-related preferences.

User message: {user_message}

Return ONLY the search query, nothing else. Keep it under 50 words."""

    response = llm.invoke([HumanMessage(content=query_prompt)])
    semantic_query = response.content.strip()
    logger.info(f"Generated semantic query: {semantic_query}")
    return semantic_query


def retrieve_user_preferences(actor_id: str, user_message: str) -> str | None:
    """Retrieve user preferences from long-term memory.

    Args:
        actor_id: The user/actor identifier.
        user_message: The user's input message.

    Returns:
        Formatted preference context or None if no preferences found.
    """
    store = get_memory_store()
    if not store or not PREFERENCE_STRATEGY_ID:
        logger.info("Memory store or preference strategy not configured")
        return None

    try:
        # Generate semantic search query using LLM
        search_query = generate_preference_query(user_message)

        # Search preferences namespace: /strategies/{memoryStrategyId}/actors/{actorId}
        preferences_namespace = (
            "strategies",
            PREFERENCE_STRATEGY_ID,
            "actors",
            actor_id,
        )
        results = store.search(preferences_namespace, query=search_query, limit=5)

        logger.info(f"AgentCore memory results: {len(results) if results else 0} items")

        if not results:
            return None

        # Format preferences for context
        pref_items = []
        for item in results:
            if hasattr(item, "value") and item.value:
                content = item.value.get("content", str(item.value))
                pref_items.append(content)
                logger.info(f"  - Preference: {content[:100]}...")

        if pref_items:
            return "User Preferences from past conversations:\n" + "\n".join(
                f"- {pref}" for pref in pref_items
            )
        return None
    except Exception as e:
        logger.error(f"Error retrieving preferences: {e}")
        return None


def save_message_to_store(
    actor_id: str, session_id: str, message: HumanMessage | AIMessage
) -> None:
    """Save a message to the memory store for long-term extraction.

    Args:
        actor_id: The user/actor identifier.
        session_id: The session/thread identifier.
        message: The message to save.
    """
    store = get_memory_store()
    if not store:
        return

    try:
        namespace = (actor_id, session_id)
        store.put(namespace, str(uuid.uuid4()), {"message": message})
    except Exception:
        pass  # Silently fail - don't break the conversation


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
    fpl_team_id: str | None = None,
) -> AgentResponse:
    """Run the agent with a user message.

    Args:
        user_message: The user's input message.
        model_name: The model to use via OpenRouter.
        actor_id: User/actor identifier for memory persistence.
        session_id: Session/thread identifier for conversation continuity.
        fpl_team_id: Optional FPL team ID for personalized context.

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

    # Build system prompt with current context
    system_prompt = build_system_prompt(fpl_team_id=fpl_team_id)

    # Retrieve user preferences from long-term memory (pre-processing)
    user_preferences = None
    if actor_id:
        user_preferences = retrieve_user_preferences(
            actor_id=actor_id,
            user_message=user_message,
        )

    # Add preferences to system prompt if available
    if user_preferences:
        system_prompt = (
            f"{system_prompt}\n\n--- Long-Term Memory ---\n{user_preferences}"
        )

    # Include system message and user message
    messages = [
        ("system", system_prompt),
        ("user", user_message),
    ]

    # Save user message to store for long-term extraction
    if actor_id and session_id:
        save_message_to_store(
            actor_id=actor_id,
            session_id=session_id,
            message=HumanMessage(content=user_message),
        )

    result = agent.invoke({"messages": messages}, config=config)

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

    # Save AI response to store for long-term extraction (post-processing)
    if actor_id and session_id and final_content:
        save_message_to_store(
            actor_id=actor_id,
            session_id=session_id,
            message=AIMessage(content=final_content),
        )

    return AgentResponse(content=final_content, tool_calls=tool_calls)


if __name__ == "__main__":
    response = run_agent("What is 5 + 3?")
    print(f"Response: {response.content}")
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}({tc.args}) -> {tc.result}")
