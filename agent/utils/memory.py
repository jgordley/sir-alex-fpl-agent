"""Memory utilities for Sir Alex agent."""

import logging
import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

logger = logging.getLogger(__name__)

# Preference strategy ID for long-term memory retrieval
PREFERENCE_STRATEGY_ID = os.getenv("AGENTCORE_PREFERENCE_STRATEGY_ID")


def get_checkpointer() -> AgentCoreMemorySaver | None:
    """Get the AgentCore Memory checkpointer if configured.

    Returns:
        AgentCoreMemorySaver instance or None if not configured.
    """
    memory_id = os.getenv("AGENTCORE_MEMORY_ID")
    if not memory_id:
        return None
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    return AgentCoreMemorySaver(memory_id, region_name=region)


def get_memory_store() -> AgentCoreMemoryStore | None:
    """Get the AgentCore Memory store for long-term memory.

    Returns:
        AgentCoreMemoryStore instance or None if not configured.
    """
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
