from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decay import run_global_decay_recurrence
from .mla import apply_rope_to_tokens
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb, build_rope_cache


class LayerHybridCache(tuple):
    """
    Tuple-compatible per-layer KV/state cache that explicitly tracks `seq_len`
    even when `n_recall_heads == 0` (where `latent_kv` has 0 feature/time columns)
    and flags whether `rope_k` is already RoPE-rotated (`rope_is_rotated=True`).

    Subclasses `tuple` so `isinstance(cache, tuple)` and tuple unpacking
    `(global_state, latent_kv)` / `(global_state, latent_kv, rope_k)` remain
    100% backward-compatible.
    """

    seq_len: int
    rope_is_rotated: bool

    def __new__(
        cls,
        items: Tuple[Any, ...],
        seq_len: int,
        rope_is_rotated: bool = True,
    ) -> "LayerHybridCache":
        obj = super().__new__(cls, items)
        obj.seq_len = int(seq_len)
        obj.rope_is_rotated = bool(rope_is_rotated)
        return obj


def _extract_past_seq_len(
    past_key_values: Optional[Tuple[Any, ...]],
    past_latent_kv: Optional[torch.Tensor],
) -> int:
    if past_key_values is None:
        return 0
    explicit_len = getattr(past_key_values, "seq_len", None)
    if explicit_len is not None:
        return int(explicit_len)
    if past_latent_kv is not None:
        return int(past_latent_kv.shape[1])
    return 0


class KilatAttention(nn.Module):
    """
    Hybrid attention combining linear-time Global Decay heads with DeepSeek-style
    Multi-head Latent Attention (MLA) recall heads.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        use_fusion_gate: bool = False,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.n_recall_heads = int(n_head * recall_ratio)
        self.n_global_heads = n_head - self.n_recall_heads
        self.latent_dim = latent_dim if latent_dim is not None else (n_embd // 4)
        self.q_lora_rank = q_lora_rank
        self.use_fusion_gate = use_fusion_gate
        self.attn_drop = attn_drop

        # Path 1: Global Decay heads
        self.log_lambda: Optional[nn.Parameter]
        if self.n_global_heads > 0:
            self.log_lambda = nn.Parameter(torch.full((self.n_global_heads,), math.log(0.9 / 0.1)))
            self.v_proj_global: Optional[nn.Linear] = nn.Linear(
                n_embd, self.n_global_heads * self.head_dim, bias=False
            )
            self.o_proj_global: Optional[nn.Linear] = nn.Linear(
                self.n_global_heads * self.head_dim, n_embd, bias=False
            )
        else:
            self.register_parameter("log_lambda", None)
            self.v_proj_global = None
            self.o_proj_global = None

        # Path 2: Latent MLA recall heads (DeepSeek-V2-Lite alignment)
        if self.n_recall_heads > 0:
            # Query path: Direct (q_lora_rank is None, DeepSeek-V2-Lite standard) or Low-Rank
            if self.q_lora_rank is None:
                self.q_proj: Optional[nn.Linear] = nn.Linear(
                    n_embd, self.n_recall_heads * self.head_dim, bias=False
                )
                self.q_a_proj: Optional[nn.Linear] = None
                self.q_a_norm: Optional[nn.Module] = None
                self.q_b_proj: Optional[nn.Linear] = None
            else:
                self.q_proj = None
                self.q_a_proj = nn.Linear(n_embd, self.q_lora_rank, bias=False)
                self.q_a_norm = RMSNorm(self.q_lora_rank)
                self.q_b_proj = nn.Linear(
                    self.q_lora_rank, self.n_recall_heads * self.head_dim, bias=False
                )

            # KV compression path (DeepSeek Low-Rank KV Joint Compression)
            self.kv_a_proj: Optional[nn.Linear] = nn.Linear(n_embd, self.latent_dim, bias=False)
            self.kv_a_norm: Optional[nn.Module] = RMSNorm(self.latent_dim)
            self.kv_b_proj: Optional[nn.Linear] = nn.Linear(
                self.latent_dim, 2 * self.n_recall_heads * self.head_dim, bias=False
            )
            self.o_proj_recall: Optional[nn.Linear] = nn.Linear(
                self.n_recall_heads * self.head_dim, n_embd, bias=False
            )
        else:
            self.q_proj = None
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = None
            self.kv_a_proj = None
            self.kv_a_norm = None
            self.kv_b_proj = None
            self.o_proj_recall = None

        # Mechanistic synergy balance gate: learnable scalar balancing Global Decay and MLA Recall
        if self.n_global_heads > 0 and self.n_recall_heads > 0:
            self.synergy_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("synergy_gate", None)

        # Optional fusion gate (gamma_net)
        if self.use_fusion_gate:
            self.gamma_net: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(2 * n_embd, n_embd),
                nn.ReLU(),
                nn.Linear(n_embd, n_embd),
                nn.Sigmoid(),
            )
        else:
            self.gamma_net = None

    @property
    def c_proj(self) -> nn.Linear:
        """Backward-compatible alias for primary residual output projection."""
        if self.o_proj_recall is not None:
            return self.o_proj_recall
        if self.o_proj_global is not None:
            return self.o_proj_global
        raise RuntimeError("No active output projection found in attention layer.")

    def _run_global_decay_path(
        self,
        x: torch.Tensor,
        past_global_state: Optional[torch.Tensor],
        past_seq_len: int,
        has_cache: bool,
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.n_global_heads == 0 or self.log_lambda is None or self.v_proj_global is None:
            B, N, _ = x.shape
            Dh = self.head_dim
            out_global = torch.zeros(B, N, 0, device=x.device, dtype=x.dtype)
            new_state = torch.zeros(B, 0, Dh, device=x.device, dtype=torch.float32) if use_cache else None
            return out_global, new_state

        V_global_flat = self.v_proj_global(x)
        return run_global_decay_recurrence(
            v_global_flat=V_global_flat,
            log_lambda=self.log_lambda,
            n_global_heads=self.n_global_heads,
            head_dim=self.head_dim,
            past_global_state=past_global_state,
            past_seq_len=past_seq_len,
            has_cache=has_cache,
            use_cache=use_cache,
            out_dtype=x.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, LayerHybridCache]]:
        B, N, _ = x.shape
        Dh = self.head_dim

        has_cache = past_key_values is not None
        if has_cache and past_key_values is not None:
            past_global_state = past_key_values[0]
            past_latent_kv = past_key_values[1]
        else:
            past_global_state, past_latent_kv = None, None

        past_seq_len = _extract_past_seq_len(past_key_values, past_latent_kv)
        total_seq_len = past_seq_len + N

        # Path 1: Global Decay
        out_global, new_global_state = self._run_global_decay_path(
            x=x,
            past_global_state=past_global_state,
            past_seq_len=past_seq_len,
            has_cache=has_cache,
            use_cache=use_cache,
        )

        # Path 2: Latent MLA
        new_latent_kv: Optional[torch.Tensor] = None
        has_q = (self.q_proj is not None) or (
            self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
        )
        if (
            self.n_recall_heads == 0
            or not has_q
            or self.kv_a_proj is None
            or self.kv_a_norm is None
            or self.kv_b_proj is None
        ):
            out_recall = torch.zeros(B, N, 0, device=x.device, dtype=x.dtype)
            if use_cache:
                new_latent_kv = torch.zeros(B, 0, self.latent_dim, device=x.device, dtype=x.dtype)
        else:
            if self.q_proj is not None:
                Q_rec = self.q_proj(x).view(B, N, self.n_recall_heads, Dh).transpose(1, 2)
            else:
                assert self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
                q_latent = self.q_a_norm(self.q_a_proj(x))
                Q_rec = self.q_b_proj(q_latent).view(B, N, self.n_recall_heads, Dh).transpose(1, 2)

            kv_latent = self.kv_a_norm(self.kv_a_proj(x))
            if has_cache and past_latent_kv is not None and past_latent_kv.shape[1] > 0:
                kv_latent_combined = torch.cat([past_latent_kv, kv_latent], dim=1)
            else:
                kv_latent_combined = kv_latent

            if use_cache:
                new_latent_kv = kv_latent_combined.clone()

            kv_full = self.kv_b_proj(kv_latent_combined)
            KV_rec = kv_full.view(B, -1, 2, self.n_recall_heads, Dh).permute(2, 0, 3, 1, 4)
            K_rec, V_rec = KV_rec[0], KV_rec[1]
            kv_len = K_rec.shape[2]

            if not has_cache or past_seq_len == 0:
                attn_mask = None
                is_causal = N > 1
            elif N == 1:
                attn_mask = None
                is_causal = False
            else:
                # Chunked prefill with cached prefix: query j attends to keys 0 .. past_seq_len + j
                attn_mask = torch.ones((N, kv_len), dtype=torch.bool, device=x.device).tril(diagonal=past_seq_len)
                is_causal = False

            out_recall = F.scaled_dot_product_attention(
                Q_rec,
                K_rec,
                V_rec,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=is_causal,
            )
            out_recall = out_recall.transpose(1, 2).reshape(B, N, self.n_recall_heads * Dh)

        # Mechanistic Decoupled Projections & Synergy Gating
        h_global = self.o_proj_global(out_global) if (self.n_global_heads > 0 and self.o_proj_global is not None) else None
        h_recall = self.o_proj_recall(out_recall) if (self.n_recall_heads > 0 and self.o_proj_recall is not None) else None

        if h_recall is not None and h_global is not None and self.synergy_gate is not None:
            gate = torch.sigmoid(self.synergy_gate)
            output = gate * h_recall + (1.0 - gate) * h_global
        elif h_recall is not None:
            output = h_recall
        elif h_global is not None:
            output = h_global
        else:
            output = torch.zeros(B, N, self.n_embd, device=x.device, dtype=x.dtype)

        if self.gamma_net is not None:
            gamma = self.gamma_net(torch.cat([x, output], dim=-1))
            output = output * gamma

        if use_cache:
            cache_obj = LayerHybridCache(
                (new_global_state, new_latent_kv),
                seq_len=total_seq_len,
                rope_is_rotated=True,
            )
            return output, cache_obj
        return output


class KilatAttentionRoPE(KilatAttention):
    """
    Hybrid attention with Decoupled Rotary Position Embedding (RoPE) on the MLA recall path.
    Caches post-RoPE keys so incremental decoding never re-rotates historical keys.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        q_lora_rank: Optional[int] = None,
        use_fusion_gate: bool = False,
        attn_drop: float = 0.0,
        rope_head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        yarn_scale: Optional[float] = None,
        yarn_base_seq_len: int = 4096,
    ) -> None:
        super().__init__(
            n_embd=n_embd,
            n_head=n_head,
            recall_ratio=recall_ratio,
            latent_dim=latent_dim,
            q_lora_rank=q_lora_rank,
            use_fusion_gate=use_fusion_gate,
            attn_drop=attn_drop,
        )

        self.rope_base = rope_base
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else (self.head_dim // 2)
        if self.rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {self.rope_head_dim}")

        self.yarn_scale = yarn_scale
        self.yarn_base_seq_len = yarn_base_seq_len
        if self.yarn_scale is not None and self.yarn_scale > 1.0:
            self.yarn_attn_factor = math.sqrt(0.1 * math.log(self.yarn_scale) + 1.0)
        else:
            self.yarn_attn_factor = 1.0

        if self.n_recall_heads > 0:
            if self.q_lora_rank is None:
                self.q_rope_proj: Optional[nn.Linear] = nn.Linear(
                    n_embd,
                    self.n_recall_heads * self.rope_head_dim,
                    bias=False,
                )
            else:
                self.q_rope_proj = nn.Linear(
                    self.q_lora_rank,
                    self.n_recall_heads * self.rope_head_dim,
                    bias=False,
                )
            self.k_rope_proj: Optional[nn.Linear] = nn.Linear(
                n_embd,
                self.rope_head_dim,
                bias=False,
            )
        else:
            self.q_rope_proj = None
            self.k_rope_proj = None

    def _apply_rope_to_tensor(
        self,
        t: torch.Tensor,
        seq_offset: int,
        rope_dim: int,
    ) -> torch.Tensor:
        return apply_rope_to_tokens(
            t=t,
            seq_offset=seq_offset,
            rope_dim=rope_dim,
            rope_base=self.rope_base,
            yarn_scale=self.yarn_scale,
            yarn_base_seq_len=self.yarn_base_seq_len,
        )

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple[Any, ...]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, LayerHybridCache]]:
        B, N, _ = x.shape
        Dh = self.head_dim
        Hr = self.n_recall_heads
        Rd = self.rope_head_dim

        has_cache = past_key_values is not None
        if has_cache and past_key_values is not None:
            past_global_state = past_key_values[0]
            past_latent_kv = past_key_values[1]
            past_rope_k = past_key_values[2] if len(past_key_values) > 2 else None
        else:
            past_global_state = past_latent_kv = past_rope_k = None

        past_seq_len = _extract_past_seq_len(past_key_values, past_latent_kv)
        total_seq_len = past_seq_len + N

        # Path 1: Global Decay
        out_global, new_global_state = self._run_global_decay_path(
            x=x,
            past_global_state=past_global_state,
            past_seq_len=past_seq_len,
            has_cache=has_cache,
            use_cache=use_cache,
        )

        # Path 2: Latent MLA + Decoupled RoPE
        new_latent_kv: Optional[torch.Tensor] = None
        new_rope_k: Optional[torch.Tensor] = None

        has_q = (self.q_proj is not None) or (
            self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
        )
        if (
            Hr == 0
            or not has_q
            or self.kv_a_proj is None
            or self.kv_a_norm is None
            or self.kv_b_proj is None
            or self.q_rope_proj is None
            or self.k_rope_proj is None
        ):
            out_recall = torch.zeros(B, N, 0, device=x.device, dtype=x.dtype)
            if use_cache:
                new_latent_kv = torch.zeros(B, 0, self.latent_dim, device=x.device, dtype=x.dtype)
                new_rope_k = torch.zeros(B, 0, Rd, device=x.device, dtype=x.dtype)
        else:
            if self.q_proj is not None:
                Q_sem = self.q_proj(x).view(B, N, Hr, Dh).transpose(1, 2)
                Q_rope = self.q_rope_proj(x).view(B, N, Hr, Rd).transpose(1, 2)
            else:
                assert self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
                q_latent = self.q_a_norm(self.q_a_proj(x))
                Q_sem = self.q_b_proj(q_latent).view(B, N, Hr, Dh).transpose(1, 2)
                Q_rope = self.q_rope_proj(q_latent).view(B, N, Hr, Rd).transpose(1, 2)

            kv_latent = self.kv_a_norm(self.kv_a_proj(x))
            k_rope_raw_new = self.k_rope_proj(x)  # (B, N, Rd)

            # Rotate ONLY the new tokens at positions [past_seq_len .. past_seq_len + N - 1]
            Q_rope_rotated = self._apply_rope_to_tensor(Q_rope, past_seq_len, Rd)
            k_rope_new_rotated = self._apply_rope_to_tensor(
                k_rope_raw_new.unsqueeze(1),
                past_seq_len,
                Rd,
            ).squeeze(1)

            if has_cache and past_latent_kv is not None and past_latent_kv.shape[1] > 0:
                kv_latent_combined = torch.cat([past_latent_kv, kv_latent], dim=1)
                if past_rope_k is not None:
                    rope_already_rotated = getattr(past_key_values, "rope_is_rotated", False)
                    if not rope_already_rotated:
                        past_rope_rot = self._apply_rope_to_tensor(past_rope_k.unsqueeze(1), 0, Rd).squeeze(1)
                    else:
                        past_rope_rot = past_rope_k
                    k_rope_rotated_shared = torch.cat([past_rope_rot, k_rope_new_rotated], dim=1)
                else:
                    k_rope_rotated_shared = k_rope_new_rotated
            else:
                kv_latent_combined = kv_latent
                k_rope_rotated_shared = k_rope_new_rotated

            if use_cache:
                new_latent_kv = kv_latent_combined.clone()
                new_rope_k = k_rope_rotated_shared.clone()

            kv_len = kv_latent_combined.shape[1]
            K_rope_rotated = k_rope_rotated_shared.unsqueeze(1).expand(B, Hr, kv_len, Rd)

            kv_full = self.kv_b_proj(kv_latent_combined)
            KV_rec = kv_full.view(B, kv_len, 2, Hr, Dh).permute(2, 0, 3, 1, 4)
            K_sem, V_rec = KV_rec[0], KV_rec[1]

            scale = math.sqrt(Dh + Rd) * getattr(self, "yarn_attn_factor", 1.0)
            score_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            score_sem = torch.matmul(Q_sem.to(score_dtype), K_sem.to(score_dtype).transpose(-2, -1))
            score_rope = torch.matmul(
                Q_rope_rotated.to(score_dtype),
                K_rope_rotated.to(score_dtype).transpose(-2, -1),
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

        # Mechanistic Decoupled Projections & Synergy Gating
        h_global = self.o_proj_global(out_global) if (self.n_global_heads > 0 and self.o_proj_global is not None) else None
        h_recall = self.o_proj_recall(out_recall) if (self.n_recall_heads > 0 and self.o_proj_recall is not None) else None

        if h_recall is not None and h_global is not None and self.synergy_gate is not None:
            gate = torch.sigmoid(self.synergy_gate)
            output = gate * h_recall + (1.0 - gate) * h_global
        elif h_recall is not None:
            output = h_recall
        elif h_global is not None:
            output = h_global
        else:
            output = torch.zeros(B, N, self.n_embd, device=x.device, dtype=x.dtype)

        if self.gamma_net is not None:
            gamma = self.gamma_net(torch.cat([x, output], dim=-1))
            output = output * gamma

        if use_cache:
            cache_obj = LayerHybridCache(
                (new_global_state, new_latent_kv, new_rope_k),
                seq_len=total_seq_len,
                rope_is_rotated=True,
            )
            return output, cache_obj
        return output

    def get_cache_size_info(self, seq_len: int) -> dict[str, Union[int, float]]:
        info: dict[str, Union[int, float]] = {
            "global_state": self.n_global_heads * self.head_dim,
            "latent_kv": seq_len * self.latent_dim if self.n_recall_heads > 0 else 0,
            "rope_k": seq_len * self.rope_head_dim if self.n_recall_heads > 0 else 0,
        }
        total = int(info["global_state"]) + int(info["latent_kv"]) + int(info["rope_k"])
        info["total_elements"] = total
        info["total_bytes_approx"] = total * 2
        return info
