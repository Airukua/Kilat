"""Assurance tests for Phase 1: fused_recall_attention (Gate 2 & Gate 3)."""

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from kilat.models.mla import LatentAttentionLayer, absorbed_recall_decode, fused_recall_attention
from kilat.models.transformer import KilatConfig, KilatTransformer
from tests.shared.bdd import And, Given, Scenario, Then, When


def _reference_recall_attention(
    q_sem: torch.Tensor,
    k_sem: torch.Tensor,
    v: torch.Tensor,
    *,
    q_rope: Optional[torch.Tensor],
    k_rope: Optional[torch.Tensor],
    scale: float,
    past_seq_len: int,
) -> torch.Tensor:
    """Explicit float64 reference implementation matching the original math."""
    score_dtype = torch.float64
    score_sem = torch.matmul(q_sem.to(score_dtype), k_sem.to(score_dtype).transpose(-2, -1))
    if q_rope is not None and k_rope is not None:
        score_rope = torch.matmul(q_rope.to(score_dtype), k_rope.to(score_dtype).transpose(-2, -1))
        scores = (score_sem + score_rope) / scale
    else:
        scores = score_sem / scale
    n_query, kv_len = q_sem.shape[2], k_sem.shape[2]
    if n_query > 1:
        causal_mask = torch.triu(
            torch.full((n_query, kv_len), float("-inf"), device=q_sem.device, dtype=score_dtype),
            diagonal=1 + past_seq_len,
        )
        scores = scores + causal_mask
    attn = F.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(attn, v)


@pytest.mark.parametrize(
    "n_query,past_seq_len,rope_dim,yarn_factor",
    [
        (16, 0, 16, 1.0),   # prefill
        (1, 12, 16, 1.0),   # single-token decode
        (5, 7, 16, 1.0),    # chunked prefill
        (12, 0, 16, 1.35),  # YaRN scaling
        (10, 0, 0, 1.0),    # no-RoPE
    ],
)
def test_fused_recall_attention_gate3_equivalence(
    n_query: int,
    past_seq_len: int,
    rope_dim: int,
    yarn_factor: float,
) -> None:
    with Scenario("Gate 3: fused_recall_attention matches explicit FP64 reference"):
        with Given("random semantic and decoupled-RoPE Q/K/V tensors"):
            torch.manual_seed(42)
            bsz, heads, head_dim = 2, 4, 32
            kv_len = past_seq_len + n_query
            q_sem = torch.randn(bsz, heads, n_query, head_dim, dtype=torch.float64)
            k_sem = torch.randn(bsz, heads, kv_len, head_dim, dtype=torch.float64)
            v = torch.randn(bsz, heads, kv_len, head_dim, dtype=torch.float64)
            q_rope = torch.randn(bsz, heads, n_query, rope_dim, dtype=torch.float64) if rope_dim > 0 else None
            k_rope = torch.randn(bsz, heads, kv_len, rope_dim, dtype=torch.float64) if rope_dim > 0 else None
            scale = math.sqrt(head_dim + rope_dim) * yarn_factor

            with When("comparing fused_recall_attention and _fused_recall_attention against FP64 reference"):
                ref = _reference_recall_attention(
                    q_sem, k_sem, v, q_rope=q_rope, k_rope=k_rope, scale=scale, past_seq_len=past_seq_len
                )
                fused_64 = fused_recall_attention(
                    q_sem, k_sem, v, q_rope=q_rope, k_rope=k_rope, scale=scale, past_seq_len=past_seq_len
                )
                fused_32 = fused_recall_attention(
                    q_sem.float(),
                    k_sem.float(),
                    v.float(),
                    q_rope=q_rope.float() if q_rope is not None else None,
                    k_rope=k_rope.float() if k_rope is not None else None,
                    scale=scale,
                    past_seq_len=past_seq_len,
                )

                with Then("FP64 matches within 1e-10 and FP32 matches within 1e-5"):
                    assert (fused_64 - ref).abs().max().item() <= 1e-10
                    assert (fused_32 - ref.float()).abs().max().item() <= 1e-5


def test_fused_recall_attention_gate2_causal_and_batch_isolation() -> None:
    with Scenario("Gate 2: Causal future-blindness, batch isolation, and gradient equivalence"):
        with Given("an FP64 prefill batch requiring gradients"):
            torch.manual_seed(7)
            bsz, heads, seq_len, head_dim, rope_dim = 2, 2, 6, 16, 8
            q_sem = torch.randn(bsz, heads, seq_len, head_dim, dtype=torch.float64, requires_grad=True)
            k_sem = torch.randn(bsz, heads, seq_len, head_dim, dtype=torch.float64, requires_grad=True)
            v = torch.randn(bsz, heads, seq_len, head_dim, dtype=torch.float64, requires_grad=True)
            q_rope = torch.randn(bsz, heads, seq_len, rope_dim, dtype=torch.float64, requires_grad=True)
            k_rope = torch.randn(bsz, heads, seq_len, rope_dim, dtype=torch.float64, requires_grad=True)
            scale = math.sqrt(head_dim + rope_dim)

            with When("checking backward gradient against future tokens"):
                out = fused_recall_attention(
                    q_sem, k_sem, v, q_rope=q_rope, k_rope=k_rope, scale=scale, past_seq_len=0
                )
                out[0, :, 2, :].sum().backward()

                with Then("future positions t > 2 have exact zero gradient"):
                    assert k_sem.grad is not None and v.grad is not None and k_rope.grad is not None
                    assert k_sem.grad[0, :, 3:, :].abs().max().item() == 0.0
                    assert v.grad[0, :, 3:, :].abs().max().item() == 0.0
                    assert k_rope.grad[0, :, 3:, :].abs().max().item() == 0.0

                with And("sample b=1 receives zero gradient from sample b=0 loss"):
                    assert q_sem.grad is not None
                    assert q_sem.grad[1].abs().max().item() == 0.0
                    assert k_sem.grad[1].abs().max().item() == 0.0
                    assert v.grad[1].abs().max().item() == 0.0


def test_latent_attention_layer_backward_and_cache_parity() -> None:
    with Scenario("Unit: LatentAttentionLayer with fused_recall_attention preserves cache parity and gradients"):
        with Given("a LatentAttentionLayer with RoPE enabled"):
            torch.manual_seed(11)
            layer = LatentAttentionLayer(n_embd=64, n_head=4, latent_dim=16, rope_head_dim=16, use_rope=True)
            layer.eval()
            x = torch.randn(1, 7, 64)

            with When("comparing full prefill vs 1-by-1 decode"):
                out_full, _, _ = layer(x, use_cache=True)
                past_kv, past_rope = None, None
                outs = []
                for t in range(x.shape[1]):
                    step_out, past_kv, past_rope = layer(
                        x[:, t : t + 1, :],
                        past_latent_kv=past_kv,
                        past_rope_k=past_rope,
                        past_seq_len=t,
                        use_cache=True,
                    )
                    outs.append(step_out)
                out_step = torch.cat(outs, dim=1)

                with Then("outputs match within 1e-5 and backward produces finite gradients"):
                    assert (out_full - out_step).abs().max().item() <= 1e-5
                    layer.train()
                    out_train, _, _ = layer(x, use_cache=False)
                    out_train.sum().backward()
                    for name, p in layer.named_parameters():
                        assert p.grad is not None, f"missing grad for {name}"
                        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


@pytest.mark.parametrize("rope_dim", [0, 16])
def test_absorbed_recall_decode_gate3_equivalence(rope_dim: int) -> None:
    with Scenario("Gate 3: absorbed_recall_decode matches explicit kv_b_proj decode"):
        with Given("a latent KV cache and kv_b_proj weight"):
            torch.manual_seed(99)
            bsz, heads, kv_len, head_dim, latent_dim = 2, 4, 15, 24, 16
            q_sem = torch.randn(bsz, heads, 1, head_dim, dtype=torch.float64)
            kv_latent = torch.randn(bsz, kv_len, latent_dim, dtype=torch.float64)
            kv_b_weight = torch.randn(2 * heads * head_dim, latent_dim, dtype=torch.float64)
            q_rope = torch.randn(bsz, heads, 1, rope_dim, dtype=torch.float64) if rope_dim > 0 else None
            k_rope = torch.randn(bsz, heads, kv_len, rope_dim, dtype=torch.float64) if rope_dim > 0 else None
            scale = math.sqrt(head_dim + rope_dim)

            with When("comparing absorbed_recall_decode against explicit kv_b_proj expansion"):
                kv_full = F.linear(kv_latent, kv_b_weight).view(bsz, kv_len, 2, heads, head_dim)
                k_sem = kv_full[:, :, 0].transpose(1, 2)
                v_rec = kv_full[:, :, 1].transpose(1, 2)
                ref = _reference_recall_attention(
                    q_sem,
                    k_sem,
                    v_rec,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    scale=scale,
                    past_seq_len=kv_len - 1,
                )
                abs_64 = absorbed_recall_decode(
                    q_sem,
                    kv_latent,
                    kv_b_weight,
                    q_rope=q_rope,
                    k_rope=k_rope,
                    scale=scale,
                    past_seq_len=kv_len - 1,
                )
                abs_32 = absorbed_recall_decode(
                    q_sem.float(),
                    kv_latent.float(),
                    kv_b_weight.float(),
                    q_rope=q_rope.float() if q_rope is not None else None,
                    k_rope=k_rope.float() if k_rope is not None else None,
                    scale=scale,
                    past_seq_len=kv_len - 1,
                )

                with Then("absorbed decode matches explicit FP64 within 1e-10 and FP32 within 1e-5"):
                    assert (abs_64 - ref).abs().max().item() <= 1e-10
                    assert (abs_32 - ref.float()).abs().max().item() <= 1e-5


@pytest.mark.parametrize("use_rope", [False, True])
def test_transformer_checkpoint_strict_and_absorbed_decode_parity(use_rope: bool) -> None:
    with Scenario("Gate 3/6: KilatTransformer strict checkpoint load and absorbed decode parity"):
        with Given("a small KilatTransformer model"):
            torch.manual_seed(123)
            cfg = KilatConfig(
                vocab_size=128,
                n_embd=64,
                n_layer=2,
                n_head=4,
                latent_dim=16,
                block_size=64,
                ffn_mode="dense",
                use_rope=use_rope,
            )
            model = KilatTransformer(cfg)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            reloaded = KilatTransformer(cfg)
            reloaded.load_state_dict(state, strict=True)
            reloaded.eval()

            with When("comparing full-sequence prefill vs prefill+absorbed multi-step decode"):
                ids = torch.randint(0, cfg.vocab_size, (1, 10))
                logits_full = reloaded(ids, use_cache=False).logits

                prefill_out = reloaded(ids[:, :4], use_cache=True)
                cache = prefill_out.past_key_values
                step_logits = [prefill_out.logits]
                for pos in range(4, 10):
                    step_out = reloaded(ids[:, pos : pos + 1], past_key_values=cache, use_cache=True)
                    cache = step_out.past_key_values
                    step_logits.append(step_out.logits)
                logits_decode = torch.cat(step_logits, dim=1)

                with Then("strict state_dict loads cleanly and decode logits match prefill within 1e-4"):
                    assert (logits_full - logits_decode).abs().max().item() <= 1e-4
