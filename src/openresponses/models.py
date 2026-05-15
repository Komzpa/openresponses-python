"""
Open Responses Data Models.

This module defines the standardized data structures for the Open Responses protocol.
It uses Pydantic V2 for strict validation and schema generation.
"""

from typing import List, Optional, Union, Literal, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

# --- STANDARDIZATION 1: Atomic Items ---
# Unlike legacy APIs (which used "messages"), Open Responses uses typed "Items"
# for both input and output. This allows mixing text, images, tools, and reasoning.


class InputText(BaseModel):
    """Represents a text input block."""

    type: Literal["input_text"] = "input_text"
    text: str


class MessageItem(BaseModel):
    """Represents a standard chat message."""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[InputText]]


# --- STANDARDIZATION 2: First-Class Reasoning ---
# The spec explicitly defines reasoning as a visible item, not a hidden "thought".
class ReasoningItem(BaseModel):
    """Represents a block of reasoning/thought process."""

    type: Literal["reasoning"] = "reasoning"
    content: Optional[str] = Field(
        default=None, description="Raw reasoning traces (Transparent)"
    )
    summary: Optional[str] = Field(default=None, description="Sanitized summary")
    encrypted_content: Optional[str] = Field(
        default=None, description="Provider-secure reasoning"
    )


class ToolCallItem(BaseModel):
    """Represents a tool call request."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: Dict[str, Any]


# Union of all possible output items
ResponseItem = Union[MessageItem, ReasoningItem, ToolCallItem]


# --- STANDARDIZATION 3: The Request Body ---
# Supports "Agentic" fields like max_tool_calls (provider-managed loops).
class OpenResponsesRequest(BaseModel):
    """Standard Request Body for Open Responses API."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, List[MessageItem], List[Dict[str, Any]]]
    stream: bool = False
    max_tool_calls: Optional[int] = Field(
        default=None, description="Limit for provider-managed loops"
    )
    max_output_tokens: Optional[int] = Field(
        default=None, description="Limit for generated output tokens"
    )
    store: Optional[bool] = Field(default=None, description="Provider storage opt-in")
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tool definitions made available to the model"
    )
    user: Optional[str] = Field(
        default=None, description="End-user identifier for provider policy controls"
    )
    text: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured output or text generation options"
    )


# --- STANDARDIZATION 4: The Response Body ---
# Output is a list of Items, not "choices".
class OpenResponsesOutput(BaseModel):
    """Standard Response Body for Open Responses API."""

    id: str
    object: Literal["response"] = "response"
    created: int
    model: str
    output: List[ResponseItem]


# --- STANDARDIZATION 5: Semantic Streaming Events ---
# Events are specific to the item type (e.g., reasoning vs text).
class StreamEvent(BaseModel):
    """Event structure for semantic streaming."""

    event: str  # e.g., 'response.reasoning.delta', 'response.text.delta'
    data: Dict[str, Any]
