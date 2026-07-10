"""Sir Alex LangGraph Agent - Graph construction and execution."""

import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from agent.utils import (
    AgentState,
    AgentResponse,
    ToolCall,
    get_checkpointer,
    retrieve_user_preferences,
    save_message_to_store,
    build_system_prompt,
    create_agent_node,
    create_tool_node,
    classify_user_query,
    ON_SOCCER_TOPIC,
    DO_NOT_ANSWER,
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sigil_client = None


def _sigil_disabled() -> bool:
    return os.getenv("SIGIL_DISABLED", "").strip().lower() in ("1", "true", "yes")


def _get_sigil_client():
    global sigil_client
    if sigil_client is None:
        from sigil_sdk import Client

        sigil_client = Client()
    return sigil_client


def _with_sigil_config(
    config: dict | None,
    *,
    user_message: str,
    model_name: str,
    actor_id: str | None,
    session_id: str | None,
) -> dict | None:
    if _sigil_disabled():
        return config

    try:
        from sigil_sdk_langgraph import SigilLangGraphHandler

        title = " ".join(user_message.split())[:120] or "Sir Alex FPL turn"
        metadata = {
            "sir_alex.model_name": model_name,
        }
        if actor_id:
            metadata["sir_alex.actor_id"] = actor_id
        if session_id:
            metadata["sir_alex.session_id"] = session_id
            metadata["conversation_id"] = session_id

        merged = dict(config or {})
        merged["metadata"] = {**dict(merged.get("metadata") or {}), **metadata}
        existing = merged.get("callbacks")
        if isinstance(existing, list):
            callbacks = list(existing)
        elif existing is None:
            callbacks = []
        else:
            callbacks = [existing]

        if not any(
            isinstance(callback, SigilLangGraphHandler) for callback in callbacks
        ):
            callbacks.append(
                SigilLangGraphHandler(
                    client=_get_sigil_client(),
                    provider_resolver="auto",
                    agent_name=os.getenv("SIGIL_AGENT_NAME", "sir-alex-fpl"),
                    agent_version=os.getenv("SIGIL_AGENT_VERSION", "1.0.0"),
                    capture_workflow_steps=True,
                    conversation_title=title,
                    extra_metadata=metadata,
                )
            )
        merged["callbacks"] = callbacks
        return merged
    except Exception as exc:
        logger.warning("Sigil LangGraph instrumentation disabled for this run: %s", exc)
        return config


def create_agent(
    model_name: str = "anthropic/claude-haiku-4.5",
    checkpointer=None,
    llm: Any | None = None,
):
    """Create the Sir Alex agent graph.

    Args:
        model_name: The model to use via OpenRouter.
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled LangGraph agent.
    """
    agent_node = create_agent_node(model_name, llm=llm)
    tool_node = create_tool_node()

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
    thread_id: str | None = None,
    fpl_team_id: str | None = None,
    llm: Any | None = None,
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
    if session_id is None:
        session_id = thread_id

    # Guardrail: Check if the query is related to soccer/FPL
    classification = classify_user_query(user_message)
    if classification == DO_NOT_ANSWER:
        return AgentResponse(
            content="I'm Sir Alex Ferguson, and I'm here to help you with Fantasy Premier League and football matters. Please ask me something related to soccer, the Premier League, or FPL!",
            tool_calls=[],
        )

    configurable = {}
    if session_id:
        configurable["thread_id"] = session_id
    if actor_id:
        configurable["actor_id"] = actor_id

    config = {"configurable": configurable} if configurable else None

    # Only use checkpointer if actor_id and session_id are provided
    checkpointer = None
    existing_message_count = 0
    if actor_id and session_id:
        checkpointer = get_checkpointer()

    agent = create_agent(model_name, checkpointer=checkpointer, llm=llm)

    # Build config for memory if checkpointer is available
    if checkpointer:
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

    config = _with_sigil_config(
        config,
        user_message=user_message,
        model_name=model_name,
        actor_id=actor_id,
        session_id=session_id,
    )
    result = agent.invoke({"messages": messages}, config=config)

    # Extract tool calls and their results from NEW messages only
    tool_calls = []
    messages = result["messages"]
    new_messages = messages[existing_message_count:]  # Only process new messages

    for i, message in enumerate(new_messages):
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Find the corresponding tool result by tool_call_id
                tool_result = ""
                tc_id = tc.get("id")
                for next_msg in new_messages[i + 1 :]:
                    if (
                        hasattr(next_msg, "tool_call_id")
                        and next_msg.tool_call_id == tc_id
                    ):
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
