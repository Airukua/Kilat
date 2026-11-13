"""BDD tests for standalone GlobalDecayLayer and LatentAttentionLayer."""
import pytest
import torch

from kilat.models.decay import GlobalDecayLayer, run_global_decay_recurrence
from kilat.models.mla import LatentAttentionLayer, apply_rope_to_tokens
from kilat.models.attention import KilatAttention, KilatAttentionRoPE

_PARAM_PREFIX = "Parameter "


class TestDecoupledLayers:
    def test_global_decay_layer_forward_and_cache_parity(self):
        """GlobalDecayLayer prefill and incremental decode must match analytically."""
        torch.manual_seed(42)
        B, N, D, H = 2, 8, 64, 4
        layer = GlobalDecayLayer(n_embd=D, n_head=H)

        x = torch.randn(B, N, D, dtype=torch.float32)

        # Full prefill pass
        out_full, final_state = layer(x, use_cache=True)
        assert out_full.shape == (B, N, D)
        assert final_state is not None
        assert final_state.shape == (B, H, D // H)

        # Step-by-step autoregressive decode
        acc_state = None
        step_outputs = []
        for t in range(N):
            token = x[:, t : t + 1, :]
            out_t, acc_state = layer(
                token,
                past_state=acc_state,
                past_seq_len=t,
                use_cache=True,
            )
            step_outputs.append(out_t)

        out_step = torch.cat(step_outputs, dim=1)
        torch.testing.assert_close(out_full, out_step, rtol=1e-5, atol=1e-5)

    def test_global_decay_layer_backward_gradients(self):
        """All parameters in GlobalDecayLayer must receive finite, non-zero gradients."""
        B, N, D, H = 2, 4, 32, 2
        layer = GlobalDecayLayer(n_embd=D, n_head=H)
        x = torch.randn(B, N, D, requires_grad=True)

        out, _ = layer(x)
        loss = out.sum()
        loss.backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, f"{_PARAM_PREFIX}{name} has no gradient"
            assert torch.isfinite(p.grad).all(), f"{_PARAM_PREFIX}{name} gradient is not finite"
            assert p.grad.abs().sum().item() > 0.0, f"{_PARAM_PREFIX}{name} gradient is dead (zero)"

    def test_latent_attention_layer_forward_and_rope(self):
        """LatentAttentionLayer must support low-rank KV compression and Decoupled RoPE."""
        torch.manual_seed(42)
        B, N, D, H = 2, 6, 64, 4
        latent_dim = 32
        rope_head_dim = 16

        mla_rope = LatentAttentionLayer(
            n_embd=D,
            n_head=H,
            latent_dim=latent_dim,
            use_rope=True,
            rope_head_dim=rope_head_dim,
        )

        x = torch.randn(B, N, D)
        out, new_latent_kv, new_rope_k = mla_rope(x, use_cache=True)

        assert out.shape == (B, N, D)
        assert new_latent_kv is not None and new_latent_kv.shape == (B, N, latent_dim)
        assert new_rope_k is not None and new_rope_k.shape == (B, N, rope_head_dim)

        loss = out.sum()
        loss.backward()

        for name, p in mla_rope.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{_PARAM_PREFIX}{name} has no gradient"
                assert torch.isfinite(p.grad).all(), f"{_PARAM_PREFIX}{name} gradient is not finite"

    def test_kilat_attention_state_dict_exact_invariants(self):
        """KilatAttention and KilatAttentionRoPE must maintain 100% parameter name backward compatibility."""
        attn_hybrid = KilatAttention(n_embd=64, n_head=4, recall_ratio=0.5)
        sd_hybrid = set(attn_hybrid.state_dict().keys())
        expected_hybrid = {
            "log_lambda",
            "v_proj_global.weight",
            "o_proj_global.weight",
            "q_proj.weight",
            "kv_a_proj.weight",
            "kv_a_norm.weight",
            "kv_b_proj.weight",
            "o_proj_recall.weight",
            "synergy_gate",
        }
        assert expected_hybrid.issubset(sd_hybrid)

        attn_rope = KilatAttentionRoPE(n_embd=64, n_head=4, recall_ratio=0.5, rope_head_dim=8)
        sd_rope = set(attn_rope.state_dict().keys())
        expected_rope = expected_hybrid.union({"q_rope_proj.weight", "k_rope_proj.weight"})
        assert expected_rope.issubset(sd_rope)
