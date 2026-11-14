"""
Kilat Serving Integration Bridge.

Provides an OpenAI-compatible REST API server with real-time Server-Sent Events (SSE)
token-by-token streaming, easily connectable to Web UIs (Open-WebUI, LibreChat, Chatbox).
"""

from __future__ import annotations

from .schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from .openai_server import create_app

__all__ = [
    "create_app",
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
]
