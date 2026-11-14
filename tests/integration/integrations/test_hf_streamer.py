"""
Tests for Hugging Face TextStreamer compatibility in GenerationMixin.
"""

from __future__ import annotations

import torch
import pytest

from kilat.configs import KilatConfig
from kilat.models import KilatTransformer
from kilat.integrations.huggingface.modeling_kilat import KilatForCausalLM


class DummyStreamer:
    """Mock streamer matching Hugging Face BaseStreamer protocol."""

    def __init__(self) -> None:
        self.tokens: list[int] = []
        self.ended = False

    def put(self, value: torch.Tensor) -> None:
        # value is a tensor of token ids
        self.tokens.extend(value.flatten().tolist())

    def end(self) -> None:
        self.ended = True


def test_generation_mixin_streamer_greedy() -> None:
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

    input_ids = torch.tensor([[1, 10, 20]], dtype=torch.long)
    streamer = DummyStreamer()

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
        streamer=streamer,
    )

    assert len(streamer.tokens) > 0
    assert len(streamer.tokens) <= 5
    assert streamer.ended is True
    # Output length equals prompt + streamed tokens
    assert out.shape[1] == input_ids.shape[1] + len(streamer.tokens)


def test_generation_mixin_streamer_sample() -> None:
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

    input_ids = torch.tensor([[1, 15, 25]], dtype=torch.long)
    streamer = DummyStreamer()

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=4,
        do_sample=True,
        temperature=0.8,
        streamer=streamer,
    )

    assert len(streamer.tokens) > 0
    assert len(streamer.tokens) <= 4
    assert streamer.ended is True
    assert out.shape[1] == input_ids.shape[1] + len(streamer.tokens)
