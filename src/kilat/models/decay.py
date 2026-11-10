"""Global Exponential Decay Attention primitives for Kilat.

Implements linear-time O(N) associative decay recurrence with Triton CUDA kernel
acceleration and exact analytical CPU fallbacks.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .kernels.flash_decay import compute_decay_normalizer_from_logit, triton_global_decay


def run_global_decay_recurrence(
    v_global_flat: torch.Tensor,
    log_lambda: torch.Tensor,
    n_global_heads: int,
    head_dim: int,
    past_global_state: Optional[torch.Tensor] = None,
    past_seq_len: int = 0,
    has_cache: bool = False,
    use_cache: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Execute the O(N) linear-time exponential decay recurrence across heads.

    Args:
        v_global_flat: Projected value tensor of shape (B, N, n_global_heads * head_dim).
        log_lambda: Learnable unconstrained log-decay parameters of shape (n_global_heads,).
        n_global_heads: Number of active exponential decay heads.
        head_dim: Dimensionality of each attention head.
        past_global_state: Cached recurrent state from prior steps, shape (B, n_global_heads, head_dim).
        past_seq_len: Historical sequence length processed so far.
        has_cache: Whether past_global_state is present and active.
        use_cache: Whether to return updated recurrent state for incremental decoding.
        out_dtype: Target output tensor dtype (defaults to v_global_flat.dtype).

    Returns:
        out_global: Decayed output tensor of shape (B, N, n_global_heads * head_dim).
        new_state: Updated state tensor (B, n_global_heads, head_dim) if use_cache else None.
    """
    B, N, _ = v_global_flat.shape
    Dh = head_dim
    dtype = out_dtype or v_global_flat.dtype

    if n_global_heads == 0:
        out_global = torch.zeros(B, N, 0, device=v_global_flat.device, dtype=dtype)
        new_state = (
            torch.zeros(B, 0, Dh, device=v_global_flat.device, dtype=torch.float32)
            if use_cache
            else None
        )
        return out_global, new_state

    compute_dtype = torch.float64 if v_global_flat.dtype == torch.float64 else torch.float32
    lam = torch.sigmoid(log_lambda.to(compute_dtype))  # (n_global_heads,)
    V_global = v_global_flat.view(B, N, n_global_heads, Dh).transpose(1, 2).contiguous()

    # Path A: Stateful incremental decoding or cached prefill continuation
    if has_cache and past_global_state is not None:
        lam_bc = lam.view(1, n_global_heads, 1)
        acc_state = past_global_state.to(compute_dtype)
        V_up = V_global.to(compute_dtype)
        step_outputs = []
        for j in range(N):
            acc_state = lam_bc * acc_state + V_up[:, :, j, :]
            z_j = compute_decay_normalizer_from_logit(
                log_lambda.to(compute_dtype),
                past_seq_len + j,
            ).view(1, n_global_heads, 1)
            step_outputs.append((acc_state / z_j).to(dtype))

        out_global = (
            torch.stack(step_outputs, dim=2)
            .transpose(1, 2)
            .reshape(B, N, n_global_heads * Dh)
        )
        new_state = acc_state if use_cache else None
        return out_global, new_state

    # Path B: Full parallel prefix scan via Triton kernel (CUDA) or analytical fallback (CPU)
    out_decay = triton_global_decay(
        lam.to(V_global.dtype) if V_global.dtype == torch.float64 else lam,
        V_global,
    )
    out_global = out_decay.transpose(1, 2).reshape(B, N, n_global_heads * Dh)

    new_state = None
    if use_cache:
        lam_bc = lam.view(1, n_global_heads, 1)
        V_up = V_global.to(compute_dtype)
        acc_state = torch.zeros(B, n_global_heads, Dh, device=v_global_flat.device, dtype=compute_dtype)
        for t in range(N):
            acc_state = lam_bc * acc_state + V_up[:, :, t, :]
        new_state = acc_state

    return out_global, new_state


class GlobalDecayLayer(nn.Module):
    """Standalone modular Sub-Quadratic Global Exponential Decay Layer.

    Encapsulates input projection (v_proj), learnable decay rate (log_lambda),
    recurrent scan execution, and output projection (o_proj).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = head_dim if head_dim is not None else (n_embd // n_head)

        self.log_lambda = nn.Parameter(torch.full((n_head,), math.log(0.9 / 0.1)))
        self.v_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_head * self.head_dim, n_embd, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
        past_seq_len: int = 0,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute forward pass through global decay layer.

        Args:
            x: Input embeddings of shape (B, N, n_embd).
            past_state: Optional prior step accumulator of shape (B, n_head, head_dim).
            past_seq_len: Length of prefix already consumed.
            use_cache: If True, returns (output, next_state) for autoregressive decode.

        Returns:
            output: Projected output of shape (B, N, n_embd).
            new_state: Updated accumulator tensor if use_cache is True.
        """
        v_flat = self.v_proj(x)
        out_decay, new_state = run_global_decay_recurrence(
            v_global_flat=v_flat,
            log_lambda=self.log_lambda,
            n_global_heads=self.n_head,
            head_dim=self.head_dim,
            past_global_state=past_state,
            past_seq_len=past_seq_len,
            has_cache=past_state is not None,
            use_cache=use_cache,
            out_dtype=x.dtype,
        )
        output = self.o_proj(out_decay)
        return output, new_state
