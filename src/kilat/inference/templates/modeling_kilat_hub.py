"""
# ==============================================================================
# DISTRIBUTION ARTIFACT FOR HUGGING FACE HUB REMOTE CODE
# ==============================================================================
# This file is a self-contained distribution template deployed by `kilat export-hf`
# for remote execution on Hugging Face Hub (trust_remote_code=True).
#
# ZERO internal dependencies on `kilat.*`.
# Depends ONLY on standard libraries, `torch`, and `transformers`.
#
# DO NOT EDIT DIRECTLY AS SOURCE CODE.
# Canonical Source of Truth: `src/kilat/models/` and `src/kilat/integrations/huggingface/`
# Adheres to Senior Software Architecture Standard (Pillar 7 & 8).
# ==============================================================================
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_kilat import KilatConfig

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# ============================================================================
# 1. Decay Mathematics & Kernel Scan Primitives
# ============================================================================

def _phi1(w: torch.Tensor) -> torch.Tensor:
    """Taylor series expansion for (exp(w) - 1) / w near w = 0."""
    small = w.abs() < 1e-4
    w_safe = torch.where(small, torch.ones_like(w), w)
    exact = torch.expm1(w_safe) / w_safe
    taylor = 1.0 + w * (0.5 + w * (1.0 / 6.0 + w / 24.0))
    return torch.where(small, taylor, exact)


def compute_decay_normalizer_from_logit(
    log_lambda: torch.Tensor,
    step_pos: Union[int, torch.Tensor],
) -> torch.Tensor:
    """Exact geometric normalizer z_t(lambda) = sum_{k=0}^{t} lambda^k."""
    compute_dtype = torch.float64 if log_lambda.dtype == torch.float64 else torch.float32
    u = F.logsigmoid(log_lambda.to(compute_dtype))
    if isinstance(step_pos, int):
        n = float(step_pos + 1)
        return n * (_phi1(n * u) / _phi1(u))
    n_t = step_pos.to(device=u.device, dtype=compute_dtype) + 1.0
    return n_t * (_phi1(n_t * u) / _phi1(u))


def _cpu_global_decay(lam: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch singularity-free O(N) recurrence for global exponential decay."""
    B, H, N, D = V.shape
    if V.numel() == 0:
        return V * (lam.sum() * 0.0 + 1.0)

    compute_dtype = torch.float64 if (V.dtype == torch.float64 or lam.dtype == torch.float64) else torch.float32
    lam_b = lam.to(compute_dtype).view(1, H, 1)
    V_up = V.to(compute_dtype)

    state = torch.zeros((B, H, D), device=V.device, dtype=compute_dtype)
    z = torch.zeros((1, H, 1), device=V.device, dtype=compute_dtype)
    outputs = []
    for t in range(N):
        state = lam_b * state + V_up[:, :, t, :]
        z = lam_b * z + 1.0
        outputs.append(state / z)

    out_up = torch.stack(outputs, dim=2)
    return out_up.to(V.dtype)


def _pytorch_decay_fwd_with_aux(
    lam: torch.Tensor,
    V: torch.Tensor,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, N, D = V.shape
    lam_h = lam.to(compute_dtype)
    lam_b = lam_h.view(1, H, 1)
    V_up = V.to(compute_dtype)

    state = torch.zeros((B, H, D), device=V.device, dtype=compute_dtype)
    z_cur = torch.zeros((H,), device=V.device, dtype=compute_dtype)
    zp_cur = torch.zeros((H,), device=V.device, dtype=compute_dtype)

    y_list = []
    z_list = []
    zp_list = []
    for t in range(N):
        zp_cur = lam_h * zp_cur + z_cur
        z_cur = lam_h * z_cur + 1.0
        state = lam_b * state + V_up[:, :, t, :]
        y_list.append(state / z_cur.view(1, H, 1))
        z_list.append(z_cur)
        zp_list.append(zp_cur)

    y_up = torch.stack(y_list, dim=2)
    z_mat = torch.stack(z_list, dim=1)
    zp_mat = torch.stack(zp_list, dim=1)
    return y_up, z_mat, zp_mat


def _pytorch_decay_bwd(
    grad_out: torch.Tensor,
    y_saved: torch.Tensor,
    lam: torch.Tensor,
    z_mat: torch.Tensor,
    zp_mat: torch.Tensor,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, N, D = y_saved.shape
    g_up = grad_out.to(compute_dtype)
    y_up = y_saved.to(compute_dtype)
    lam_h = lam.to(compute_dtype)
    lam_b = lam_h.view(1, H, 1)

    r = torch.zeros((B, H, D), device=y_saved.device, dtype=compute_dtype)
    grad_lam = torch.zeros((H,), device=y_saved.device, dtype=compute_dtype)
    grad_v_rev = []

    for t in range(N - 1, -1, -1):
        z_t = z_mat[:, t].view(1, H, 1)
        g_tilde = g_up[:, :, t, :] / z_t
        r = g_tilde + lam_b * r
        grad_v_rev.append(r)

        if t > 0:
            z_prev = z_mat[:, t - 1]
            zp_t = zp_mat[:, t]
            dot_ry = (r * y_up[:, :, t - 1, :]).sum(dim=(0, 2))
            dot_gy = (g_tilde * y_up[:, :, t, :]).sum(dim=(0, 2))
            grad_lam = grad_lam + (z_prev * dot_ry - zp_t * dot_gy)

    grad_v_rev.reverse()
    grad_v = torch.stack(grad_v_rev, dim=2)
    return grad_lam, grad_v


if HAS_TRITON:
    @triton.jit
    def _flash_decay_fwd_kernel(
        v_ptr, lam_ptr, out_ptr, z_ptr, zp_ptr,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        B, H, N, D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_d = tl.program_id(1)
        b_idx = pid_bh // H
        h_idx = pid_bh % H
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh
        out_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh
        lam = tl.load(lam_ptr + h_idx).to(tl.float32)

        s = tl.zeros((BLOCK_D,), dtype=tl.float32)
        z = 0.0
        zp = 0.0

        for t in range(0, N):
            v_t = tl.load(v_base + t * stride_vn + offs_d * stride_vd, mask=d_mask, other=0.0).to(tl.float32)
            zp = lam * zp + z
            z = lam * z + 1.0
            s = lam * s + v_t
            y_t = s / z

            tl.store(out_base + t * stride_on + offs_d * stride_od, y_t, mask=d_mask)
            if pid_d == 0:
                tl.store(z_ptr + h_idx * N + t, z)
                tl.store(zp_ptr + h_idx * N + t, zp)

    @triton.jit
    def _flash_decay_bwd_kernel(
        grad_out_ptr, y_ptr, lam_ptr, z_ptr, zp_ptr,
        grad_v_ptr, grad_lam_partial_ptr,
        stride_gb, stride_gh, stride_gn, stride_gd,
        stride_yb, stride_yh, stride_yn, stride_yd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        H, N, D: tl.constexpr, NUM_D_BLOCKS: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_d = tl.program_id(1)
        b_idx = pid_bh // H
        h_idx = pid_bh % H
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        g_base = grad_out_ptr + b_idx * stride_gb + h_idx * stride_gh
        y_base = y_ptr + b_idx * stride_yb + h_idx * stride_yh
        gv_base = grad_v_ptr + b_idx * stride_vb + h_idx * stride_vh
        lam = tl.load(lam_ptr + h_idx).to(tl.float32)

        r = tl.zeros((BLOCK_D,), dtype=tl.float32)
        grad_lam_acc = 0.0

        for step in range(0, N):
            t = N - 1 - step
            g_t = tl.load(g_base + t * stride_gn + offs_d * stride_gd, mask=d_mask, other=0.0).to(tl.float32)
            y_t = tl.load(y_base + t * stride_yn + offs_d * stride_yd, mask=d_mask, other=0.0).to(tl.float32)
            z_t = tl.load(z_ptr + h_idx * N + t).to(tl.float32)

            g_tilde = g_t / z_t
            r = g_tilde + lam * r
            tl.store(gv_base + t * stride_vn + offs_d * stride_vd, r, mask=d_mask)

            if t > 0:
                y_prev = tl.load(y_base + (t - 1) * stride_yn + offs_d * stride_yd, mask=d_mask, other=0.0).to(tl.float32)
                z_prev = tl.load(z_ptr + h_idx * N + (t - 1)).to(tl.float32)
                zp_t = tl.load(zp_ptr + h_idx * N + t).to(tl.float32)
                dot_ry = tl.sum(r * y_prev, axis=0)
                dot_gy = tl.sum(g_tilde * y_t, axis=0)
                grad_lam_acc += z_prev * dot_ry - zp_t * dot_gy

        tl.store(grad_lam_partial_ptr + pid_bh * NUM_D_BLOCKS + pid_d, grad_lam_acc)
else:
    _flash_decay_fwd_kernel = None
    _flash_decay_bwd_kernel = None


class FlashDecayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam_scores: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = V.shape
        ctx.input_lam_dtype = lam_scores.dtype
        ctx.input_v_dtype = V.dtype

        if V.numel() == 0:
            ctx.empty = True
            ctx.lam_shape = lam_scores.shape
            ctx.lam_device = lam_scores.device
            return V.clone()

        ctx.empty = False
        is_fp64 = V.dtype == torch.float64 or lam_scores.dtype == torch.float64
        use_triton = bool(V.is_cuda and HAS_TRITON and not is_fp64)
        ctx.use_triton = use_triton

        if use_triton:
            V_contig = V.contiguous()
            lam_f32 = lam_scores.detach().to(torch.float32).contiguous()
            y_f32 = torch.empty((B, H, N, D), device=V.device, dtype=torch.float32)
            z_mat = torch.empty((H, N), device=V.device, dtype=torch.float32)
            zp_mat = torch.empty((H, N), device=V.device, dtype=torch.float32)

            BLOCK_D = min(128, triton.next_power_of_2(max(16, D)))
            grid = (B * H, triton.cdiv(D, BLOCK_D))
            _flash_decay_fwd_kernel[grid](
                V_contig, lam_f32, y_f32, z_mat, zp_mat,
                V_contig.stride(0), V_contig.stride(1), V_contig.stride(2), V_contig.stride(3),
                y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
                B, H, N, D=D, BLOCK_D=BLOCK_D,
            )
            ctx.save_for_backward(y_f32, lam_f32, z_mat, zp_mat)
            ctx.block_d = BLOCK_D
            return y_f32.to(V.dtype)

        compute_dtype = torch.float64 if is_fp64 else torch.float32
        ctx.compute_dtype = compute_dtype
        y_up, z_mat, zp_mat = _pytorch_decay_fwd_with_aux(lam_scores, V, compute_dtype)
        ctx.save_for_backward(y_up, lam_scores.detach().to(compute_dtype), z_mat, zp_mat)
        return y_up.to(V.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if ctx.empty:
            grad_lam = torch.zeros(ctx.lam_shape, device=ctx.lam_device, dtype=ctx.input_lam_dtype)
            return grad_lam, grad_out.to(ctx.input_v_dtype)

        y_saved, lam_saved, z_mat, zp_mat = ctx.saved_tensors
        B, H, N, D = y_saved.shape

        if ctx.use_triton and grad_out.is_cuda and HAS_TRITON:
            g_contig = grad_out.contiguous()
            grad_v_f32 = torch.empty((B, H, N, D), device=grad_out.device, dtype=torch.float32)
            BLOCK_D = ctx.block_d
            num_d_blocks = triton.cdiv(D, BLOCK_D)
            grad_lam_partial = torch.empty((B * H, num_d_blocks), device=grad_out.device, dtype=torch.float32)

            grid = (B * H, num_d_blocks)
            _flash_decay_bwd_kernel[grid](
                g_contig, y_saved, lam_saved, z_mat, zp_mat,
                grad_v_f32, grad_lam_partial,
                g_contig.stride(0), g_contig.stride(1), g_contig.stride(2), g_contig.stride(3),
                y_saved.stride(0), y_saved.stride(1), y_saved.stride(2), y_saved.stride(3),
                grad_v_f32.stride(0), grad_v_f32.stride(1), grad_v_f32.stride(2), grad_v_f32.stride(3),
                H, N, D=D, NUM_D_BLOCKS=num_d_blocks, BLOCK_D=BLOCK_D,
            )
            grad_lam = grad_lam_partial.view(B, H, num_d_blocks).sum(dim=(0, 2))
            return grad_lam.to(ctx.input_lam_dtype), grad_v_f32.to(ctx.input_v_dtype)

        compute_dtype = getattr(ctx, "compute_dtype", torch.float32)
        grad_lam, grad_v = _pytorch_decay_bwd(grad_out, y_saved, lam_saved, z_mat, zp_mat, compute_dtype)
        return grad_lam.to(ctx.input_lam_dtype), grad_v.to(ctx.input_v_dtype)


def triton_global_decay(lam_scores: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return FlashDecayFunction.apply(lam_scores, V)


# ============================================================================
# 2. Positional Embeddings & Normalization
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_up = x.to(compute_dtype)
        variance = x_up.pow(2).mean(dim=-1, keepdim=True)
        x_normed = (x_up * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return x_normed * self.weight


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def build_rope_cache(
    seq_len: int,
    rope_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    base: float = 10000.0,
    offset: int = 0,
    scale: float = 1.0,
    original_max_position_embeddings: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    half_dim = rope_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    if scale > 1.0:
        wavelength = 2.0 * math.pi / theta
        r = original_max_position_embeddings / wavelength
        gamma = torch.clamp((r - 1.0) / max(1e-6, 31.0), 0.0, 1.0)
        theta = (1.0 - gamma) * (theta / scale) + gamma * theta

    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, theta)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


# ============================================================================
# 3. Cache Structure
# ============================================================================

class LayerHybridCache(tuple):
    seq_len: int
    rope_is_rotated: bool

    def __new__(
        cls,
        items: Tuple[Any, ...],
        seq_len: int,
        rope_is_rotated: bool = True,
    ) -> LayerHybridCache:
        obj = super().__new__(cls, items)
        obj.seq_len = int(seq_len)
        obj.rope_is_rotated = bool(rope_is_rotated)
        return obj


# ============================================================================
# 4. Attention (Macro-Synergy: Decay + Latent MLA)
# ============================================================================

class KilatAttention(nn.Module):
    def __init__(
        self,
        config: KilatConfig,
        layer_recall_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        recall_ratio = float(layer_recall_ratio if layer_recall_ratio is not None else config.recall_ratio)

        self.n_recall_heads = int(self.n_head * recall_ratio)
        self.n_global_heads = self.n_head - self.n_recall_heads
        self.latent_dim = config.latent_dim
        self.q_lora_rank = config.q_lora_rank
        self.attn_drop = config.attn_drop
        self.use_rope = config.use_rope
        self.rope_head_dim = config.rope_head_dim
        self.rope_base = config.rope_base
        self.yarn_scale = config.yarn_scale
        self.yarn_base_seq_len = config.yarn_base_seq_len

        # Global Decay Heads
        if self.n_global_heads > 0:
            self.log_lambda = nn.Parameter(torch.full((self.n_global_heads,), math.log(0.9 / 0.1)))
            self.v_proj_global = nn.Linear(self.n_embd, self.n_global_heads * self.head_dim, bias=False)
            self.o_proj_global = nn.Linear(self.n_global_heads * self.head_dim, self.n_embd, bias=False)
        else:
            self.register_parameter("log_lambda", None)
            self.v_proj_global = None
            self.o_proj_global = None

        # Latent MLA Recall Heads
        if self.n_recall_heads > 0:
            if self.q_lora_rank is None:
                self.q_proj = nn.Linear(self.n_embd, self.n_recall_heads * self.head_dim, bias=False)
                self.q_a_proj = None
                self.q_a_norm = None
                self.q_b_proj = None
            else:
                self.q_proj = None
                self.q_a_proj = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
                self.q_a_norm = RMSNorm(self.q_lora_rank)
                self.q_b_proj = nn.Linear(self.q_lora_rank, self.n_recall_heads * self.head_dim, bias=False)

            self.kv_a_proj = nn.Linear(self.n_embd, self.latent_dim, bias=False)
            self.kv_a_norm = RMSNorm(self.latent_dim)
            self.kv_b_proj = nn.Linear(self.latent_dim, 2 * self.n_recall_heads * self.head_dim, bias=False)
            self.o_proj_recall = nn.Linear(self.n_recall_heads * self.head_dim, self.n_embd, bias=False)

            if self.use_rope:
                if self.q_lora_rank is None:
                    self.q_rope_proj = nn.Linear(self.n_embd, self.n_recall_heads * self.rope_head_dim, bias=False)
                else:
                    self.q_rope_proj = nn.Linear(self.q_lora_rank, self.n_recall_heads * self.rope_head_dim, bias=False)
                self.k_rope_proj = nn.Linear(self.n_embd, self.rope_head_dim, bias=False)
            else:
                self.q_rope_proj = None
                self.k_rope_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = None
            self.kv_a_proj = None
            self.kv_a_norm = None
            self.kv_b_proj = None
            self.o_proj_recall = None
            self.q_rope_proj = None
            self.k_rope_proj = None

        if self.n_global_heads > 0 and self.n_recall_heads > 0:
            self.synergy_gate = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("synergy_gate", None)

    def _apply_rope(self, t: torch.Tensor, offset: int, rope_dim: int) -> torch.Tensor:
        cos, sin = build_rope_cache(
            seq_len=t.shape[2],
            rope_dim=rope_dim,
            device=t.device,
            dtype=t.dtype,
            base=self.rope_base,
            offset=offset,
            scale=self.yarn_scale if self.yarn_scale is not None else 1.0,
            original_max_position_embeddings=self.yarn_base_seq_len,
        )
        return apply_rotary_pos_emb(t, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerHybridCache]]:
        B, N, C = x.shape
        past_global_state = past_key_values[0] if (past_key_values and len(past_key_values) > 0) else None
        past_latent_kv = past_key_values[1] if (past_key_values and len(past_key_values) > 1) else None
        past_rope_k = past_key_values[2] if (past_key_values and len(past_key_values) > 2) else None

        past_seq_len = 0
        if past_key_values is not None:
            past_seq_len = getattr(past_key_values, "seq_len", 0)
            if past_seq_len == 0 and past_latent_kv is not None:
                past_seq_len = int(past_latent_kv.shape[1])

        current_seq_len = past_seq_len + N
        has_cache = (past_seq_len > 0)

        out_global = None
        new_global_state = None
        if self.n_global_heads > 0 and self.v_proj_global is not None:
            v_g = self.v_proj_global(x)
            Dh = self.head_dim
            compute_dtype = torch.float64 if v_g.dtype == torch.float64 else torch.float32
            lam = torch.sigmoid(self.log_lambda.to(compute_dtype))
            V_g = v_g.view(B, N, self.n_global_heads, Dh).transpose(1, 2).contiguous()

            if has_cache and past_global_state is not None:
                lam_bc = lam.view(1, self.n_global_heads, 1)
                acc = past_global_state.to(compute_dtype)
                V_up = V_g.to(compute_dtype)
                steps = []
                for j in range(N):
                    acc = lam_bc * acc + V_up[:, :, j, :]
                    z_j = compute_decay_normalizer_from_logit(self.log_lambda.to(compute_dtype), past_seq_len + j).view(1, self.n_global_heads, 1)
                    steps.append((acc / z_j).to(v_g.dtype))
                out_global = torch.stack(steps, dim=2).transpose(1, 2).reshape(B, N, self.n_global_heads * Dh)
                new_global_state = acc if use_cache else None
            else:
                out_d = triton_global_decay(lam.to(V_g.dtype) if V_g.dtype == torch.float64 else lam, V_g)
                out_global = out_d.transpose(1, 2).reshape(B, N, self.n_global_heads * Dh)
                if use_cache:
                    lam_bc = lam.view(1, self.n_global_heads, 1)
                    V_up = V_g.to(compute_dtype)
                    acc = torch.zeros(B, self.n_global_heads, Dh, device=v_g.device, dtype=compute_dtype)
                    for t in range(N):
                        acc = lam_bc * acc + V_up[:, :, t, :]
                    new_global_state = acc

        out_recall = None
        new_latent_kv = None
        new_rope_k = None
        if self.n_recall_heads > 0 and self.kv_a_proj is not None:
            Hr = self.n_recall_heads
            Dh = self.head_dim

            if self.q_proj is not None:
                Q_sem = self.q_proj(x).view(B, N, Hr, Dh).transpose(1, 2)
                Q_rope = self.q_rope_proj(x).view(B, N, Hr, self.rope_head_dim).transpose(1, 2) if self.use_rope else None
            else:
                q_latent = self.q_a_norm(self.q_a_proj(x))
                Q_sem = self.q_b_proj(q_latent).view(B, N, Hr, Dh).transpose(1, 2)
                Q_rope = self.q_rope_proj(q_latent).view(B, N, Hr, self.rope_head_dim).transpose(1, 2) if self.use_rope else None

            kv_latent = self.kv_a_norm(self.kv_a_proj(x))
            if self.use_rope:
                k_rope_raw_new = self.k_rope_proj(x)
                Q_rope_rot = self._apply_rope(Q_rope, past_seq_len, self.rope_head_dim)
                k_rope_rot_new = self._apply_rope(k_rope_raw_new.unsqueeze(1), past_seq_len, self.rope_head_dim).squeeze(1)
            else:
                Q_rope_rot = None
                k_rope_rot_new = None

            if has_cache and past_latent_kv is not None and past_latent_kv.shape[1] > 0:
                kv_combined = torch.cat([past_latent_kv, kv_latent], dim=1)
                if self.use_rope and past_rope_k is not None:
                    k_rope_comb = torch.cat([past_rope_k, k_rope_rot_new], dim=1)
                else:
                    k_rope_comb = k_rope_rot_new
            else:
                kv_combined = kv_latent
                k_rope_comb = k_rope_rot_new

            if use_cache:
                new_latent_kv = kv_combined.clone()
                new_rope_k = k_rope_comb.clone() if self.use_rope else None

            kv_len = kv_combined.shape[1]
            kv_full = self.kv_b_proj(kv_combined)
            KV_rec = kv_full.view(B, kv_len, 2, Hr, Dh).permute(2, 0, 3, 1, 4)
            K_sem, V_rec = KV_rec[0], KV_rec[1]

            score_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            score_sem = torch.matmul(Q_sem.to(score_dtype), K_sem.to(score_dtype).transpose(-2, -1))

            if self.use_rope and Q_rope_rot is not None and k_rope_comb is not None:
                K_rope_rot = k_rope_comb.unsqueeze(1).expand(B, Hr, kv_len, self.rope_head_dim)
                score_rope = torch.matmul(Q_rope_rot.to(score_dtype), K_rope_rot.to(score_dtype).transpose(-2, -1))
                scale = math.sqrt(Dh + self.rope_head_dim)
                scores = (score_sem + score_rope) / scale
            else:
                scale = math.sqrt(Dh)
                scores = score_sem / scale

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

        present_cache = None
        if use_cache:
            present_cache = LayerHybridCache(
                (new_global_state, new_latent_kv, new_rope_k),
                seq_len=current_seq_len,
                rope_is_rotated=True,
            )

        return output, present_cache


# ============================================================================
# 5. Feed-Forward Networks (SwiGLU & DeepSeekMoE)
# ============================================================================

class SwiGLU(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        hidden = max(64, ((int(dim * ff_mult) + 63) // 64) * 64)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_routed_experts: int = 8,
        num_shared_experts: int = 0,
        active_experts: int = 2,
        fine_grained_factor: int = 1,
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.01,
        device_balance_coef: float = 0.0,
        router_balance_mode: str = "aux_loss",
        renormalize_topk: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.active_experts = active_experts
        self.fine_grained_factor = fine_grained_factor
        self.total_routed_experts = num_routed_experts * fine_grained_factor
        self.effective_active_experts = active_experts * fine_grained_factor
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef
        self.router_balance_mode = router_balance_mode
        self.renormalize_topk = renormalize_topk

        expert_ff_mult = ff_mult / float(fine_grained_factor)
        self.router = nn.Linear(dim, self.total_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias",
            torch.zeros(self.total_routed_experts, dtype=torch.float32),
            persistent=(router_balance_mode == "bias_free"),
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(dim, ff_mult=ff_mult, dropout=dropout) for _ in range(num_shared_experts)]
        )
        self.routed_experts = nn.ModuleList(
            [SwiGLU(dim, ff_mult=expert_ff_mult, dropout=dropout) for _ in range(self.total_routed_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        num_tokens = x_flat.shape[0]
        N_r = self.total_routed_experts
        K_eff = self.effective_active_experts

        shared_output = torch.zeros_like(x)
        for s_expert in self.shared_experts:
            shared_output = shared_output + s_expert(x)

        router_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        router_logits = F.linear(x_flat.to(router_dtype), self.router.weight.to(router_dtype))
        router_probs = F.softmax(router_logits, dim=-1)

        selection_scores = (
            router_probs + self.expert_bias.to(router_dtype)
            if self.router_balance_mode == "bias_free"
            else router_probs
        )

        _, topk_indices = torch.topk(selection_scores, K_eff, dim=-1)
        topk_weights = router_probs.gather(dim=-1, index=topk_indices)

        if K_eff > 1 and self.renormalize_topk:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)

        flat_expert_ids = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        counts = torch.bincount(flat_expert_ids, minlength=N_r)

        routed_output = torch.zeros_like(x_flat)
        if num_tokens > 0:
            sorted_expert_ids, perm = flat_expert_ids.sort()
            sorted_token_ids = torch.div(perm, K_eff, rounding_mode="floor")
            sorted_weights = flat_weights.index_select(0, perm).to(x.dtype).unsqueeze(-1)
            sorted_inputs = x_flat.index_select(0, sorted_token_ids)

            split_sizes = counts.tolist()
            input_chunks = torch.split(sorted_inputs, split_sizes, dim=0)
            out_chunks = [
                self.routed_experts[idx](chunk)
                for idx, (chunk, cnt) in enumerate(zip(input_chunks, split_sizes))
                if cnt > 0
            ]
            if out_chunks:
                sorted_outputs = torch.cat(out_chunks, dim=0) * sorted_weights
                routed_output.index_add_(0, sorted_token_ids, sorted_outputs)

        output = (shared_output.reshape(-1, D) + routed_output).reshape(B, S, D)

        denom = float(max(1, num_tokens * K_eff))
        expert_fraction = counts.to(router_dtype) / denom
        router_prob_avg = router_probs.mean(dim=0)

        if self.router_balance_mode == "bias_free":
            total_aux_loss = (router_probs.sum() * 0.0).to(x.dtype)
        else:
            expert_balance_loss = float(N_r) * (expert_fraction * router_prob_avg).sum()
            device_balance_loss = (
                torch.var(expert_fraction, dim=0, unbiased=False) * float(N_r)
                if N_r > 1
                else torch.zeros((), device=x.device, dtype=router_dtype)
            )
            total_aux_loss = (self.aux_loss_coef * expert_balance_loss + self.device_balance_coef * device_balance_loss).to(x.dtype)

        return output, total_aux_loss


class FeedForward(nn.Module):
    def __init__(self, config: KilatConfig):
        super().__init__()
        self.mode = config.ffn_mode
        self.layer: nn.Module
        if self.mode in ("dense", "swiglu"):
            self.layer = SwiGLU(config.n_embd, ff_mult=config.ff_mult, dropout=config.ffn_dropout)
        else:
            self.layer = DeepSeekMoE(
                dim=config.n_embd,
                num_routed_experts=config.num_experts,
                num_shared_experts=config.num_shared_experts,
                active_experts=config.active_experts,
                fine_grained_factor=config.fine_grained_factor,
                ff_mult=config.ff_mult,
                dropout=config.ffn_dropout,
                aux_loss_coef=config.aux_loss_coef,
                device_balance_coef=config.device_balance_coef,
                router_balance_mode=config.router_balance_mode,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode in ("dense", "swiglu"):
            return self.layer(x), None
        return self.layer(x)


class Block(nn.Module):
    def __init__(self, config: KilatConfig, layer_recall_ratio: Optional[float] = None):
        super().__init__()
        self.rms1 = RMSNorm(config.n_embd)
        self.attn = KilatAttention(config, layer_recall_ratio=layer_recall_ratio)
        self.rms2 = RMSNorm(config.n_embd)
        self.mlp = FeedForward(config)
        self.resid_drop = nn.Dropout(config.resid_drop) if config.resid_drop > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        attn_out, present_cache = self.attn(self.rms1(x), past_key_values=past_key_values, use_cache=use_cache)
        x = x + attn_out
        mlp_out, aux_loss = self.mlp(self.rms2(x))
        x = x + self.resid_drop(mlp_out)
        return x, aux_loss, present_cache


# ============================================================================
# 6. Hugging Face Models: PreTrainedModel, KilatModel, KilatForCausalLM
# ============================================================================

class KilatPreTrainedModel(PreTrainedModel):
    config_class = KilatConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class KilatModel(KilatPreTrainedModel):
    def __init__(self, config: KilatConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_drop) if config.embd_drop > 0 else nn.Identity()

        if config.recall_pattern == "macro_synergy":
            layer_recall_ratios = [0.0 if (i % 2 == 0) else 1.0 for i in range(config.n_layer)]
        elif config.recall_pattern == "custom" and config.recall_ratios is not None:
            layer_recall_ratios = [float(r) for r in config.recall_ratios]
        else:
            layer_recall_ratios = [float(config.recall_ratio) for _ in range(config.n_layer)]

        self.layers = nn.ModuleList([
            Block(config, layer_recall_ratio=layer_recall_ratios[i])
            for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding, got {type(value)}")
        self.wte = value

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Any, ...]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        x = self.drop(self.wte(input_ids))

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values = [] if use_cache else None
        hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.layers):
            if output_hidden_states:
                hidden_states = hidden_states + (x,)
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, _, present = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    past_key_values[i],
                    use_cache,
                    use_reentrant=False,
                )
            else:
                x, _, present = block(x, past_key_values=past_key_values[i], use_cache=use_cache)

            if use_cache:
                present_key_values.append(present)

        x = self.ln_f(x)
        if output_hidden_states:
            hidden_states = hidden_states + (x,)

        if not return_dict:
            return tuple(v for v in [x, present_key_values, hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=tuple(present_key_values) if use_cache else None,
            hidden_states=hidden_states,
        )


class KilatForCausalLM(KilatPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "wte.weight"}

    def __init__(self, config: KilatConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_drop) if config.embd_drop > 0 else nn.Identity()

        if config.recall_pattern == "macro_synergy":
            layer_recall_ratios = [0.0 if (i % 2 == 0) else 1.0 for i in range(config.n_layer)]
        elif config.recall_pattern == "custom" and config.recall_ratios is not None:
            layer_recall_ratios = [float(r) for r in config.recall_ratios]
        else:
            layer_recall_ratios = [float(config.recall_ratio) for _ in range(config.n_layer)]

        self.layers = nn.ModuleList([
            Block(config, layer_recall_ratio=layer_recall_ratios[i])
            for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.wte.weight

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding, got {type(value)}")
        self.wte = value
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(new_embeddings)}")
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: Optional[int] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_first_iteration: Optional[bool] = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        if past_key_values is None:
            return None
        reordered = []
        for layer_cache in past_key_values:
            if layer_cache is None:
                reordered.append(None)
                continue
            g_state, lat_kv, rope_k = layer_cache
            new_g = g_state.index_select(0, beam_idx) if g_state is not None else None
            new_kv = lat_kv.index_select(0, beam_idx) if lat_kv is not None else None
            new_rk = rope_k.index_select(0, beam_idx) if rope_k is not None else None
            reordered.append(LayerHybridCache((new_g, new_kv, new_rk), seq_len=layer_cache.seq_len, rope_is_rotated=True))
        return tuple(reordered)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Any, ...]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        x = self.drop(self.wte(input_ids))

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values = [] if use_cache else None
        hidden_states = () if output_hidden_states else None
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        has_moe = False

        for i, block in enumerate(self.layers):
            if output_hidden_states:
                hidden_states = hidden_states + (x,)
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, layer_aux_loss, present = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    past_key_values[i],
                    use_cache,
                    use_reentrant=False,
                )
            else:
                x, layer_aux_loss, present = block(x, past_key_values=past_key_values[i], use_cache=use_cache)

            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss
                has_moe = True

            if use_cache:
                present_key_values.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
            flat_logits = shift_logits.view(-1, shift_logits.size(-1)).to(loss_dtype)
            flat_labels = shift_labels.view(-1)
            nll_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
            valid_tokens = (flat_labels != -100).sum()
            safe_denom = valid_tokens.clamp(min=1).to(loss_dtype)
            ce_loss_up = nll_sum / safe_denom

            total_loss = ce_loss_up
            if has_moe:
                total_loss = total_loss + total_aux_loss.to(loss_dtype)
            loss = total_loss if logits.dtype == torch.float64 else total_loss.to(torch.float32)

        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (tuple(present_key_values),)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(present_key_values) if use_cache else None,
            hidden_states=hidden_states,
        )
