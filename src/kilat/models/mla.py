"""Multi-head Latent Attention (MLA) primitives for Kilat.

Implements DeepSeek-style low-rank Key-Value joint compression and optional
Decoupled Rotary Positional Embeddings (RoPE).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import apply_rotary_pos_emb, build_rope_cache


def apply_rope_to_tokens(
    t: torch.Tensor,
    seq_offset: int,
    rope_dim: int,
    rope_base: float = 10000.0,
    yarn_scale: Optional[float] = None,
    yarn_base_seq_len: int = 4096,
) -> torch.Tensor:
    """Apply rotary position embedding to 4D tensor (B, H, N, Rd)."""
    _, _, N, Rd = t.shape
    if Rd != rope_dim:
        raise ValueError(f"Expected rope_dim={rope_dim}, got {Rd}")

    cos, sin = build_rope_cache(
        seq_len=N,
        rope_dim=rope_dim,
        device=t.device,
        dtype=t.dtype,
        base=rope_base,
        offset=seq_offset,
        scale=yarn_scale if yarn_scale is not None else 1.0,
        original_max_position_embeddings=yarn_base_seq_len,
    )
    return apply_rotary_pos_emb(t, cos, sin)


class LatentAttentionLayer(nn.Module):
    """Standalone modular Multi-head Latent Attention (MLA) layer.

    Compresses Key-Value cache into a shared latent space via low-rank projections,
    reducing KV memory footprint by up to 4-8x with exact full-rank expressivity.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        latent_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        use_rope: bool = False,
        rope_head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        yarn_scale: Optional[float] = None,
        yarn_base_seq_len: int = 4096,
        attn_drop: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.latent_dim = latent_dim if latent_dim is not None else (n_embd // 4)
        self.q_lora_rank = q_lora_rank
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.yarn_scale = yarn_scale
        self.yarn_base_seq_len = yarn_base_seq_len
        self.attn_drop = attn_drop

        if use_rope:
            self.rope_head_dim = (
                rope_head_dim if rope_head_dim is not None else (self.head_dim // 2)
            )
            if self.rope_head_dim % 2 != 0:
                raise ValueError(f"rope_head_dim must be even, got {self.rope_head_dim}")
            if self.yarn_scale is not None and self.yarn_scale > 1.0:
                self.yarn_attn_factor = math.sqrt(0.1 * math.log(self.yarn_scale) + 1.0)
            else:
                self.yarn_attn_factor = 1.0
        else:
            self.rope_head_dim = 0
            self.yarn_attn_factor = 1.0

        # Query path: Direct or LoRA compressed
        if self.q_lora_rank is None:
            self.q_proj: Optional[nn.Linear] = nn.Linear(
                n_embd, self.n_head * self.head_dim, bias=bias
            )
            self.q_a_proj: Optional[nn.Linear] = None
            self.q_a_norm: Optional[nn.Module] = None
            self.q_b_proj: Optional[nn.Linear] = None
            if self.use_rope:
                self.q_rope_proj: Optional[nn.Linear] = nn.Linear(
                    n_embd, self.n_head * self.rope_head_dim, bias=bias
                )
            else:
                self.q_rope_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(n_embd, self.q_lora_rank, bias=bias)
            self.q_a_norm = RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.n_head * self.head_dim, bias=bias
            )
            if self.use_rope:
                self.q_rope_proj = nn.Linear(
                    self.q_lora_rank, self.n_head * self.rope_head_dim, bias=bias
                )
            else:
                self.q_rope_proj = None

        # Key-Value path: Joint latent compression
        self.kv_a_proj = nn.Linear(n_embd, self.latent_dim, bias=bias)
        self.kv_a_norm = RMSNorm(self.latent_dim)
        self.kv_b_proj = nn.Linear(
            self.latent_dim, 2 * self.n_head * self.head_dim, bias=bias
        )
        if self.use_rope:
            self.k_rope_proj: Optional[nn.Linear] = nn.Linear(
                n_embd, self.rope_head_dim, bias=bias
            )
        else:
            self.k_rope_proj = None

        # Output projection
        self.o_proj = nn.Linear(self.n_head * self.head_dim, n_embd, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        past_latent_kv: Optional[torch.Tensor] = None,
        past_rope_k: Optional[torch.Tensor] = None,
        past_seq_len: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute forward pass of Latent MLA with optional RoPE and caching."""
        B, N, _ = x.shape
        Dh = self.head_dim
        Hr = self.n_head
        Rd = self.rope_head_dim

        # Project queries
        if self.q_proj is not None:
            Q_sem = self.q_proj(x).view(B, N, Hr, Dh).transpose(1, 2)
            Q_rope = (
                self.q_rope_proj(x).view(B, N, Hr, Rd).transpose(1, 2)
                if (self.use_rope and self.q_rope_proj is not None)
                else None
            )
        else:
            assert self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
            q_latent = self.q_a_norm(self.q_a_proj(x))
            Q_sem = self.q_b_proj(q_latent).view(B, N, Hr, Dh).transpose(1, 2)
            Q_rope = (
                self.q_rope_proj(q_latent).view(B, N, Hr, Rd).transpose(1, 2)
                if (self.use_rope and self.q_rope_proj is not None)
                else None
            )

        # Compress KV into latent space
        kv_latent = self.kv_a_norm(self.kv_a_proj(x))

        if self.use_rope and self.k_rope_proj is not None:
            k_rope_raw = self.k_rope_proj(x)
            k_rope_rot_new = apply_rope_to_tokens(
                k_rope_raw.unsqueeze(1),
                past_seq_len,
                Rd,
                self.rope_base,
                self.yarn_scale,
                self.yarn_base_seq_len,
            ).squeeze(1)
            assert Q_rope is not None
            Q_rope_rot = apply_rope_to_tokens(
                Q_rope,
                past_seq_len,
                Rd,
                self.rope_base,
                self.yarn_scale,
                self.yarn_base_seq_len,
            )
        else:
            k_rope_rot_new = None
            Q_rope_rot = None

        has_cache = past_latent_kv is not None and past_latent_kv.shape[1] > 0
        if has_cache:
            assert past_latent_kv is not None
            kv_latent_combined = torch.cat([past_latent_kv, kv_latent], dim=1)
            if self.use_rope and past_rope_k is not None:
                assert k_rope_rot_new is not None
                k_rope_combined = torch.cat([past_rope_k, k_rope_rot_new], dim=1)
            else:
                k_rope_combined = k_rope_rot_new
        else:
            kv_latent_combined = kv_latent
            k_rope_combined = k_rope_rot_new

        new_latent_kv = kv_latent_combined.clone() if use_cache else None
        new_rope_k = k_rope_combined.clone() if (use_cache and self.use_rope) else None

        kv_len = kv_latent_combined.shape[1]
        kv_full = self.kv_b_proj(kv_latent_combined)
        KV_rec = kv_full.view(B, kv_len, 2, Hr, Dh).permute(2, 0, 3, 1, 4)
        K_sem, V_rec = KV_rec[0], KV_rec[1]

        if self.use_rope and k_rope_combined is not None and Q_rope_rot is not None:
            K_rope = k_rope_combined.unsqueeze(1).expand(B, Hr, kv_len, Rd)
            scale = math.sqrt(Dh + Rd) * getattr(self, "yarn_attn_factor", 1.0)
            score_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            score_sem = torch.matmul(Q_sem.to(score_dtype), K_sem.to(score_dtype).transpose(-2, -1))
            score_rope = torch.matmul(
                Q_rope_rot.to(score_dtype),
                K_rope.to(score_dtype).transpose(-2, -1),
            )
            scores = (score_sem + score_rope) / scale
            if N > 1:
                causal_mask = torch.triu(
                    torch.full((N, kv_len), float("-inf"), device=x.device, dtype=score_dtype),
                    diagonal=1 + past_seq_len,
                )
                scores = scores + causal_mask
            attn_weights = F.softmax(scores, dim=-1).to(V_rec.dtype)
            if self.training and self.attn_drop > 0:
                attn_weights = F.dropout(attn_weights, p=self.attn_drop)
            out_recall = torch.matmul(attn_weights, V_rec).transpose(1, 2).reshape(B, N, Hr * Dh)
        else:
            if not has_cache or past_seq_len == 0:
                attn_mask = None
                is_causal = N > 1
            elif N == 1:
                attn_mask = None
                is_causal = False
            else:
                attn_mask = torch.ones((N, kv_len), dtype=torch.bool, device=x.device).tril(diagonal=past_seq_len)
                is_causal = False

            out_recall = F.scaled_dot_product_attention(
                Q_sem,
                K_sem,
                V_rec,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=is_causal,
            )
            out_recall = out_recall.transpose(1, 2).reshape(B, N, Hr * Dh)

        output = self.o_proj(out_recall)
        return output, new_latent_kv, new_rope_k
