"""State definitions for Sir Alex agent."""

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """State for the Sir Alex agent."""

    messages: Annotated[list, add_messages]


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""

    name: str
    args: dict
    result: str


class AgentResponse(BaseModel):
    """Response from the agent including tool calls."""

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
