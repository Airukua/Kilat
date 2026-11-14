"""
Tests for Kilat OpenAI-Compatible Serving Bridge (/v1/chat/completions).
"""

from __future__ import annotations

import json
from typing import Any, Optional
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from kilat.configs import KilatConfig
from kilat.models import KilatTransformer
from kilat.integrations.serving import create_app

MODEL_ID = "kilat-test-moe"


class MockTokenizer:
    """Mock tokenizer for serving tests."""

    def __init__(self) -> None:
        self.vocab_size = 128
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = False,
    ) -> Any:
        res = [1, 10, 20]
        if return_tensors == "pt":
            return torch.tensor([res], dtype=torch.long)
        return res

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.flatten().tolist()
        return " word"


@pytest.fixture
def client() -> TestClient:
    config = KilatConfig(
        vocab_size=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
    )
    model = KilatTransformer(config)
    model.eval()
    tokenizer = MockTokenizer()
    app = create_app(model, tokenizer, model_id=MODEL_ID)
    return TestClient(app)


def test_health_and_models_endpoints(client: TestClient) -> None:
    # Test /health
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # Test /v1/models
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == MODEL_ID


def test_chat_completions_non_streaming(client: TestClient) -> None:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "max_new_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "content" in body["choices"][0]["message"]


def test_chat_completions_streaming_sse(client: TestClient) -> None:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Stream me a poem."}],
        "max_new_tokens": 4,
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    lines = res.text.strip().split("\n\n")
    assert len(lines) >= 3  # Initial + tokens + terminal + [DONE]
    assert lines[-1] == "data: [DONE]"

    # Verify JSON parsability of intermediate chunks
    first_data = json.loads(lines[0].replace("data: ", ""))
    assert first_data["object"] == "chat.completion.chunk"
    assert first_data["choices"][0]["delta"]["role"] == "assistant"
