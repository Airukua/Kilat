"""
kilat.integrations.serving.schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pydantic schemas for OpenAI-compatible REST API serving.
Complies with OpenAI /v1/chat/completions request and response formats.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message in conversation history."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: Optional[str] = "kilat"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=256, alias="max_new_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class ChatMessageResponse(BaseModel):
    """Message returned in non-streaming completion."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """Choice object in non-streaming response."""

    index: int = 0
    message: ChatMessageResponse
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming response schema."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "kilat"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class ChatCompletionChunkDelta(BaseModel):
    """Incremental delta token in streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice item in streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible SSE chunk schema."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    """Model information card."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kilat"


class ModelListResponse(BaseModel):
    """List of available models."""

    object: str = "list"
    data: List[ModelCard]
