"""
Unit tests for KilatAttentionRoPE mechanism and incremental KV caching.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.arc.attention import KilatAttentionRoPE
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_attention_rope_prefill_and_generation():
    with Scenario("Unit: KilatAttentionRoPE Prefill and Incremental Auto-Regressive Step"):
        with Given("a KilatAttentionRoPE module"):
            n_embd, n_head = 64, 4
            attn = KilatAttentionRoPE(n_embd=n_embd, n_head=n_head, rope_head_dim=16)

            with When("executing forward pass in prefill mode with use_cache=True"):
                batch_size, seq_len = 2, 8
                x = torch.randn(batch_size, seq_len, n_embd)
                out, cache = attn(x, use_cache=True)

                with Then("the output preserves tensor dimensions"):
                    assert out.shape == (batch_size, seq_len, n_embd)

                with And("cache contains valid state tuples"):
                    assert isinstance(cache, tuple)
                    assert len(cache) == 3  # (global_state, latent_kv, rope_k)

            with When("executing single-token incremental step with cached past"):
                x_next = torch.randn(batch_size, 1, n_embd)
                out_next, cache_next = attn(x_next, past_key_values=cache, use_cache=True)

                with Then("single-token output is correctly generated"):
                    assert out_next.shape == (batch_size, 1, n_embd)
                    assert isinstance(cache_next, tuple)

            with When("querying cache memory sizing"):
                info = attn.get_cache_size_info(seq_len=64)
                with Then("size metadata is properly computed"):
                    assert "total_elements" in info
                    assert "total_bytes_approx" in info
                    assert info["total_bytes_approx"] > 0
