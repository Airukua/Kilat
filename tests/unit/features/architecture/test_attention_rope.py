"""
Unit tests for KilatAttentionRoPE mechanism and incremental KV caching.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch

from kilat.models.attention import KilatAttention, KilatAttentionRoPE, LayerHybridCache
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
                    assert info["total_elements"] > 0
                    assert info["total_bytes_approx"] > 0


def test_hybrid_cache_prefill_decode_and_chunked_equivalence_matrix():
    with Scenario("Unit: M-05 LayerHybridCache Prefill vs Token Decode vs Chunked Prefill Matrix"):
        with Given("KilatAttention and KilatAttentionRoPE across recall_ratio in {0.0, 0.5, 1.0}"):
            torch.manual_seed(99)
            x_full = torch.randn(2, 9, 32)

            for use_rope in (False, True):
                for recall_ratio in (0.0, 0.5, 1.0):
                    if use_rope:
                        attn = KilatAttentionRoPE(
                            n_embd=32,
                            n_head=4,
                            recall_ratio=recall_ratio,
                            rope_head_dim=8,
                            attn_drop=0.0,
                        )
                    else:
                        attn = KilatAttention(
                            n_embd=32,
                            n_head=4,
                            recall_ratio=recall_ratio,
                            attn_drop=0.0,
                        )
                    attn.eval()

                    with When(f"comparing full prefill vs 1-by-1 decode vs chunked (4+3+2) [rope={use_rope}, ratio={recall_ratio}]"):
                        with torch.no_grad():
                            out_full = attn(x_full, use_cache=False)

                            # 1-by-1 token decode
                            cache_inc = None
                            inc_parts = []
                            for step_idx in range(x_full.shape[1]):
                                step_out, cache_inc = attn(
                                    x_full[:, step_idx : step_idx + 1, :],
                                    past_key_values=cache_inc,
                                    use_cache=True,
                                )
                                inc_parts.append(step_out)
                            out_inc = torch.cat(inc_parts, dim=1)

                            # Chunked prefill: 4 + 3 + 2 tokens
                            c1, cache_chk = attn(x_full[:, :4, :], use_cache=True)
                            c2, cache_chk = attn(x_full[:, 4:7, :], past_key_values=cache_chk, use_cache=True)
                            c3, cache_chk = attn(x_full[:, 7:9, :], past_key_values=cache_chk, use_cache=True)
                            out_chunked = torch.cat([c1, c2, c3], dim=1)

                        with Then("LayerHybridCache preserves seq_len=9 and outputs match within 1e-5"):
                            assert isinstance(cache_inc, LayerHybridCache)
                            assert isinstance(cache_inc, tuple)
                            assert cache_inc.seq_len == 9
                            assert cache_chk.seq_len == 9
                            assert torch.max(torch.abs(out_full - out_inc)).item() < 1e-5
                            assert torch.max(torch.abs(out_full - out_chunked)).item() < 1e-5
