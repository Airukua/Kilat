from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels.flash_decay import compute_decay_normalizer_from_logit, triton_global_decay
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
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.n_recall_heads = int(n_head * recall_ratio)
        self.n_global_heads = n_head - self.n_recall_heads
        self.latent_dim = latent_dim if latent_dim is not None else (n_embd // 4)

        # Path 1: Global Decay heads
        self.log_lambda: Optional[nn.Parameter]
        if self.n_global_heads > 0:
            self.log_lambda = nn.Parameter(torch.full((self.n_global_heads,), math.log(0.9 / 0.1)))
            self.v_proj_global: Optional[nn.Linear] = nn.Linear(
                n_embd, self.n_global_heads * self.head_dim, bias=False
            )
        else:
            self.register_parameter("log_lambda", None)
            self.v_proj_global = None

        # Path 2: Latent MLA recall heads
        if self.n_recall_heads > 0:
            self.q_a_proj: Optional[nn.Linear] = nn.Linear(n_embd, self.latent_dim, bias=False)
            self.q_a_norm: Optional[nn.LayerNorm] = nn.LayerNorm(self.latent_dim)
            self.q_b_proj: Optional[nn.Linear] = nn.Linear(
                self.latent_dim, self.n_recall_heads * self.head_dim, bias=False
            )
            self.kv_a_proj: Optional[nn.Linear] = nn.Linear(n_embd, self.latent_dim, bias=False)
            self.kv_a_norm: Optional[nn.LayerNorm] = nn.LayerNorm(self.latent_dim)
            self.kv_b_proj: Optional[nn.Linear] = nn.Linear(
                self.latent_dim, 2 * self.n_recall_heads * self.head_dim, bias=False
            )
        else:
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = None
            self.kv_a_proj = None
            self.kv_a_norm = None
            self.kv_b_proj = None

        # Fusion gate & output projection
        self.gamma_net = nn.Sequential(
            nn.Linear(2 * n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Sigmoid(),
        )
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = attn_drop

    def _run_global_decay_path(
        self,
        x: torch.Tensor,
        past_global_state: Optional[torch.Tensor],
        past_seq_len: int,
        has_cache: bool,
        use_cache: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = x.shape
        Dh = self.head_dim

        if self.n_global_heads == 0 or self.log_lambda is None or self.v_proj_global is None:
            out_global = torch.zeros(B, N, 0, device=x.device, dtype=x.dtype)
            new_state = torch.zeros(B, 0, Dh, device=x.device, dtype=torch.float32) if use_cache else None
            return out_global, new_state

        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        lam = torch.sigmoid(self.log_lambda.to(compute_dtype))  # (H_g,)
        V_global_flat = self.v_proj_global(x)
        V_global = V_global_flat.view(B, N, self.n_global_heads, Dh).transpose(1, 2).contiguous()

        if has_cache and past_global_state is not None:
            lam_bc = lam.view(1, self.n_global_heads, 1)
            acc_state = past_global_state.to(compute_dtype)
            V_up = V_global.to(compute_dtype)
            step_outputs = []
            for j in range(N):
                acc_state = lam_bc * acc_state + V_up[:, :, j, :]
                z_j = compute_decay_normalizer_from_logit(
                    self.log_lambda.to(compute_dtype),
                    past_seq_len + j,
                ).view(1, self.n_global_heads, 1)
                step_outputs.append((acc_state / z_j).to(x.dtype))

            out_global = (
                torch.stack(step_outputs, dim=2)
                .transpose(1, 2)
                .reshape(B, N, self.n_global_heads * Dh)
            )
            new_state = acc_state if use_cache else None
            return out_global, new_state

        out_decay = triton_global_decay(lam.to(V_global.dtype) if V_global.dtype == torch.float64 else lam, V_global)
        out_global = out_decay.transpose(1, 2).reshape(B, N, self.n_global_heads * Dh)

        new_state = None
        if use_cache:
            lam_bc = lam.view(1, self.n_global_heads, 1)
            V_up = V_global.to(compute_dtype)
            acc_state = torch.zeros(B, self.n_global_heads, Dh, device=x.device, dtype=compute_dtype)
            for t in range(N):
                acc_state = lam_bc * acc_state + V_up[:, :, t, :]
            new_state = acc_state

        return out_global, new_state

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
        if (
            self.n_recall_heads == 0
            or self.q_a_proj is None
            or self.q_a_norm is None
            or self.q_b_proj is None
            or self.kv_a_proj is None
            or self.kv_a_norm is None
            or self.kv_b_proj is None
        ):
            out_recall = torch.zeros(B, N, 0, device=x.device, dtype=x.dtype)
            if use_cache:
                new_latent_kv = torch.zeros(B, 0, self.latent_dim, device=x.device, dtype=x.dtype)
        else:
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

        # Fusion & Output Projection
        out_combined = torch.cat([out_global, out_recall], dim=-1)
        gate = self.gamma_net(torch.cat([x, out_combined], dim=-1))
        output = self.c_proj(out_combined * gate)

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
        attn_drop: float = 0.0,
        rope_head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__(n_embd, n_head, recall_ratio, latent_dim, attn_drop)

        self.rope_base = rope_base
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else (self.head_dim // 2)
        if self.rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even, got {self.rope_head_dim}")

        if self.n_recall_heads > 0:
            self.q_rope_proj: Optional[nn.Linear] = nn.Linear(
                self.latent_dim,
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
        _, _, N, Rd = t.shape
        if Rd != rope_dim:
            raise ValueError(f"Expected rope_dim={rope_dim}, got {Rd}")

        cos, sin = build_rope_cache(
            seq_len=N,
            rope_dim=rope_dim,
            device=t.device,
            dtype=t.dtype,
            base=self.rope_base,
            offset=seq_offset,
        )
        return apply_rotary_pos_emb(t, cos, sin)

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

        if (
            Hr == 0
            or self.q_a_proj is None
            or self.q_a_norm is None
            or self.q_b_proj is None
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

            scale = math.sqrt(Dh + Rd)
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

        out_combined = torch.cat([out_global, out_recall], dim=-1)
        gate = self.gamma_net(torch.cat([x, out_combined], dim=-1))
        output = self.c_proj(out_combined * gate)

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
