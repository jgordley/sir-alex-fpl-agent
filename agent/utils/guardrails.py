"""Guardrails for Sir Alex agent."""

import logging
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Classification results
ON_SOCCER_TOPIC = "ON_SOCCER_TOPIC"
DO_NOT_ANSWER = "DO_NOT_ANSWER"


def classify_user_query(user_message: str) -> str:
    """Classify if a user query is related to soccer/Premier League/FPL.

    Args:
        user_message: The user's input message.

    Returns:
        ON_SOCCER_TOPIC if the query is about soccer/FPL, DO_NOT_ANSWER otherwise.
    """
    model_id = os.getenv("CONVERSATION_CLASSIFIER_MODEL_ID", "mistralai/ministral-8b")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.info("Guardrail check skipped - OPENROUTER_API_KEY is not configured")
        return ON_SOCCER_TOPIC

    llm = ChatOpenAI(
        model=model_id,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )

    classification_prompt = f"""You are a content classifier. Your job is to determine if a user message is related to soccer, football, Premier League, Fantasy Premier League (FPL), or any related topics.

Classify the following user message into EXACTLY one of these two categories:
- ON_SOCCER_TOPIC: The message is about soccer, football, Premier League, FPL, players, teams, matches, transfers, tactics, or any related topic.
- DO_NOT_ANSWER: The message is NOT related to soccer/football at all.

User message: {user_message}

Respond with ONLY one of these two values: ON_SOCCER_TOPIC or DO_NOT_ANSWER
Do not include any other text, explanation, or punctuation."""

    try:
        logger.info(f"Guardrail check - User query: {user_message}")
        response = llm.invoke([HumanMessage(content=classification_prompt)])
        classification = response.content.strip().upper()
        logger.info(f"Guardrail check - Raw model response: {classification}")

        # Normalize the response
        if ON_SOCCER_TOPIC in classification:
            result = ON_SOCCER_TOPIC
        elif DO_NOT_ANSWER in classification:
            result = DO_NOT_ANSWER
        else:
            # Default to allowing if unclear
            logger.warning(
                f"Guardrail check - Unclear classification: {classification}"
            )
            result = ON_SOCCER_TOPIC

        logger.info(f"Guardrail check - Final classification: {result}")
        return result
    except Exception as e:
        logger.error(f"Guardrail check - Error: {e}")
        # Default to allowing on error
        return ON_SOCCER_TOPIC
