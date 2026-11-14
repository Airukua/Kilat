"""
kilat.integrations.serving.openai_server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI server exposing OpenAI-compatible /v1/chat/completions endpoint
with Server-Sent Events (SSE) token-by-token streaming.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Iterator, Optional

from kilat.inference.generator import TextGenerator
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ModelCard,
    ModelListResponse,
    UsageInfo,
)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
except ImportError as exc:
    raise ImportError(
        "FastAPI is required for Kilat serving. Please run: pip install fastapi uvicorn"
    ) from exc

logger = logging.getLogger(__name__)


def _format_messages_to_prompt(tokenizer: Any, messages: list[Any]) -> str:
    """Format conversation messages into a plain-text prompt string."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.debug("Chat template application failed, falling back: %s", exc)

    # Standard fallback prompt template
    prompt_lines: list[str] = []
    for m in messages:
        if m.role == "system":
            prompt_lines.append(f"System: {m.content}\n")
        elif m.role == "user":
            prompt_lines.append(f"User: {m.content}\n")
        elif m.role == "assistant":
            prompt_lines.append(f"Assistant: {m.content}\n")
    prompt_lines.append("Assistant: ")
    return "\n".join(prompt_lines)


def create_app(
    model: Any,
    tokenizer: Any,
    model_id: str = "kilat",
) -> FastAPI:
    """
    Create an OpenAI-compatible FastAPI application.

    Parameters
    ----------
    model : KilatTransformer or KilatForCausalLM
        Loaded Kilat model instance.
    tokenizer : Any
        PreTrainedTokenizer or KilatTokenizer instance.
    model_id : str, default="kilat"
        Name of the model reported in /v1/models and completions.

    Returns
    -------
    FastAPI
        Configured web application instance.
    """
    app = FastAPI(title="Kilat OpenAI-Compatible Serving Engine", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    generator = TextGenerator(model, tokenizer)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy", "model": model_id}

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        return ModelListResponse(data=[ModelCard(id=model_id)])

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        prompt = _format_messages_to_prompt(tokenizer, request.messages)
        max_new_tokens = request.max_tokens or 256
        do_sample = request.temperature > 0.0
        temperature = max(request.temperature, 1e-4) if do_sample else 1.0

        if request.stream:
            def event_generator() -> Iterator[str]:
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())

                # Initial role chunk
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant", content=""),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"

                # Stream decoded tokens
                for token_text in generator.stream(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                ):
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created_ts,
                        model=model_id,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=token_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"

                # Terminal chunk
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming mode
        generated_text = generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        return ChatCompletionResponse(
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessageResponse(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(generated_text.split()),
                total_tokens=len(prompt.split()) + len(generated_text.split()),
            ),
        )

    return app
