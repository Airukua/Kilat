from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - fallback for platforms without Triton
    HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def _phi1(w: torch.Tensor) -> torch.Tensor:
    """
    Smooth analytic evaluation of (exp(w) - 1) / w without catastrophic cancellation.
    Uses 4th-order Taylor polynomial around w = 0 for |w| < 1e-4.
    """
    small = w.abs() < 1e-4
    w_safe = torch.where(small, torch.ones_like(w), w)
    exact = torch.expm1(w_safe) / w_safe
    taylor = 1.0 + w * (0.5 + w * (1.0 / 6.0 + w / 24.0))
    return torch.where(small, taylor, exact)


def compute_decay_normalizer_from_logit(
    log_lambda: torch.Tensor,
    step_pos: Union[int, torch.Tensor],
) -> torch.Tensor:
    """
    Compute exact geometric normalizer z_t(lambda) = sum_{k=0}^{t} lambda^k
    where lambda = sigmoid(log_lambda), stable across all log_lambda in (-inf, +inf)
    (including lambda -> 1^-) and arbitrarily large sequence positions t >= 0.

    Returns tensor in float32 (or float64 if log_lambda is float64) to prevent FP16
    overflow when t > 65504.
    """
    compute_dtype = torch.float64 if log_lambda.dtype == torch.float64 else torch.float32
    u = F.logsigmoid(log_lambda.to(compute_dtype))  # ln(lambda) <= 0
    if isinstance(step_pos, int):
        n = float(step_pos + 1)
        return n * (_phi1(n * u) / _phi1(u))
    n_t = step_pos.to(device=u.device, dtype=compute_dtype) + 1.0
    return n_t * (_phi1(n_t * u) / _phi1(u))


def _reference_global_decay_fp64(lam: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Exact O(N^2) causal matrix reference implementation in float64 for verification gates.
    M[h, i, j] = lambda_h^(i - j) / sum_{k=0}^{i} lambda_h^k  for j <= i, else 0.
    """
    B, H, N, D = V.shape
    if V.numel() == 0:
        return V.to(torch.float64).clone()

    lam64 = lam.to(torch.float64)  # (H,)
    V64 = V.to(torch.float64)  # (B, H, N, D)
    idx = torch.arange(N, device=V.device, dtype=torch.float64)
    dist = idx[:, None] - idx[None, :]  # (N, N)
    causal = idx[:, None] >= idx[None, :]  # (N, N)

    # Safe powers: for dist >= 0 and lam in [0, 1], 0^0 == 1.0 on diagonal
    lam_grid = lam64[:, None, None]  # (H, 1, 1)
    safe_dist = torch.where(causal[None, :, :], dist[None, :, :], torch.zeros_like(dist)[None, :, :])
    weights_unnorm = torch.pow(lam_grid, safe_dist) * causal[None, :, :].to(torch.float64)  # (H, N, N)
    z = weights_unnorm.sum(dim=-1, keepdim=True)  # (H, N, 1), strictly >= 1.0
    M = weights_unnorm / z  # (H, N, N)
    return torch.einsum("hij,bhjd->bhid", M, V64)


def _cpu_global_decay(lam: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch singularity-free O(N) recurrence for global exponential decay.
    Computes:
        s_t = lam * s_{t-1} + V_t
        z_t = lam * z_{t-1} + 1      (strictly >= 1.0 for all lam >= 0; exact at lam = 1.0)
        out_t = s_t / z_t
    """
    B, H, N, D = V.shape
    if V.numel() == 0:
        return V * (lam.sum() * 0.0 + 1.0)

    compute_dtype = torch.float64 if (V.dtype == torch.float64 or lam.dtype == torch.float64) else torch.float32
    lam_b = lam.to(compute_dtype).view(1, H, 1)  # (1, H, 1)
    V_up = V.to(compute_dtype)

    state = torch.zeros((B, H, D), device=V.device, dtype=compute_dtype)
    z = torch.zeros((1, H, 1), device=V.device, dtype=compute_dtype)
    outputs = []
    for t in range(N):
        state = lam_b * state + V_up[:, :, t, :]
        z = lam_b * z + 1.0
        outputs.append(state / z)

    out_up = torch.stack(outputs, dim=2)  # (B, H, N, D)
    return out_up.to(V.dtype)


def _pytorch_decay_fwd_with_aux(
    lam: torch.Tensor,
    V: torch.Tensor,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward recurrence returning (y_up, z, z_prime) in compute_dtype."""
    B, H, N, D = V.shape
    lam_h = lam.to(compute_dtype)  # (H,)
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

    y_up = torch.stack(y_list, dim=2)  # (B, H, N, D)
    z_mat = torch.stack(z_list, dim=1)  # (H, N)
    zp_mat = torch.stack(zp_list, dim=1)  # (H, N)
    return y_up, z_mat, zp_mat


def _pytorch_decay_bwd(
    grad_out: torch.Tensor,
    y_saved: torch.Tensor,
    lam: torch.Tensor,
    z_mat: torch.Tensor,
    zp_mat: torch.Tensor,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact O(N) analytical reverse-mode differentiation for global decay:
        g_tilde_t = g_t / z_t
        r_t = g_tilde_t + lam * r_{t+1}
        dL/dV_t = r_t
        dL/dlam_h = sum_{b, t>=1} (z_{h, t-1} <r_{b,h,t}, y_{b,h,t-1}> - z'_{h, t} <g_tilde_{b,h,t}, y_{b,h,t}>)
    """
    B, H, N, D = y_saved.shape
    g_up = grad_out.to(compute_dtype)
    y_up = y_saved.to(compute_dtype)
    lam_h = lam.to(compute_dtype)
    lam_b = lam_h.view(1, H, 1)

    r = torch.zeros((B, H, D), device=y_saved.device, dtype=compute_dtype)
    grad_lam = torch.zeros((H,), device=y_saved.device, dtype=compute_dtype)
    grad_v_rev = []

    for t in range(N - 1, -1, -1):
        z_t = z_mat[:, t].view(1, H, 1)  # (1, H, 1)
        g_tilde = g_up[:, :, t, :] / z_t  # (B, H, D)
        r = g_tilde + lam_b * r
        grad_v_rev.append(r)

        if t > 0:
            z_prev = z_mat[:, t - 1]  # (H,)
            zp_t = zp_mat[:, t]  # (H,)
            dot_ry = (r * y_up[:, :, t - 1, :]).sum(dim=(0, 2))  # (H,)
            dot_gy = (g_tilde * y_up[:, :, t, :]).sum(dim=(0, 2))  # (H,)
            grad_lam = grad_lam + (z_prev * dot_ry - zp_t * dot_gy)

    grad_v_rev.reverse()
    grad_v = torch.stack(grad_v_rev, dim=2)  # (B, H, N, D)
    return grad_lam, grad_v


if HAS_TRITON:

    @triton.jit
    def flash_decay_fwd_kernel(
        v_ptr,
        lam_ptr,
        out_ptr,
        z_ptr,
        zp_ptr,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        B,
        H,
        N,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Linear O(N*D) forward Triton scan kernel with singularity-free normalization:
            zp_t = lam * zp_{t-1} + z_{t-1}
            z_t  = lam * z_{t-1} + 1.0
            s_t  = lam * s_{t-1} + v_t
            y_t  = s_t / z_t
        """
        pid_bh = tl.program_id(0)
        pid_d = tl.program_id(1)

        b_idx = pid_bh // H
        h_idx = pid_bh % H
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D

        v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh
        out_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh

        lam = tl.load(lam_ptr + h_idx).to(tl.float32)
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)
        z = 0.0
        zp = 0.0
        write_aux = (b_idx == 0) & (pid_d == 0)

        for t in range(0, N):
            zp = lam * zp + z
            z = lam * z + 1.0
            v_t = tl.load(v_base + t * stride_vn + offs_d * stride_vd, mask=d_mask, other=0.0).to(tl.float32)
            state = lam * state + v_t
            y_t = state / z
            tl.store(out_base + t * stride_on + offs_d * stride_od, y_t, mask=d_mask)
            if write_aux:
                tl.store(z_ptr + h_idx * N + t, z)
                tl.store(zp_ptr + h_idx * N + t, zp)

    @triton.jit
    def flash_decay_bwd_kernel(
        grad_out_ptr,
        y_ptr,
        lam_ptr,
        z_ptr,
        zp_ptr,
        grad_v_ptr,
        grad_lam_partial_ptr,
        stride_gb,
        stride_gh,
        stride_gn,
        stride_gd,
        stride_yb,
        stride_yh,
        stride_yn,
        stride_yd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        H,
        N,
        D: tl.constexpr,
        NUM_D_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Linear O(N*D) analytical backward Triton kernel.
        Writes per-(pid_bh, pid_d) scalar partials to grad_lam_partial_ptr for deterministic host reduction.
        """
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
                y_prev = tl.load(y_base + (t - 1) * stride_yn + offs_d * stride_yd, mask=d_mask, other=0.0).to(
                    tl.float32
                )
                z_prev = tl.load(z_ptr + h_idx * N + (t - 1)).to(tl.float32)
                zp_t = tl.load(zp_ptr + h_idx * N + t).to(tl.float32)
                dot_ry = tl.sum(r * y_prev, axis=0)
                dot_gy = tl.sum(g_tilde * y_t, axis=0)
                grad_lam_acc += z_prev * dot_ry - zp_t * dot_gy

        tl.store(grad_lam_partial_ptr + pid_bh * NUM_D_BLOCKS + pid_d, grad_lam_acc)
else:  # pragma: no cover
    flash_decay_fwd_kernel = None  # type: ignore[assignment]
    flash_decay_bwd_kernel = None  # type: ignore[assignment]


class FlashDecayFunction(torch.autograd.Function):
    """
    Differentiable Global Decay operator supporting both CUDA Triton kernels and
    exact PyTorch CPU / FP64 execution with identical analytical gradients.
    """

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
            # Keep full-precision forward output for accurate adjoint dot-products in backward
            y_f32 = torch.empty((B, H, N, D), device=V.device, dtype=torch.float32)
            z_mat = torch.empty((H, N), device=V.device, dtype=torch.float32)
            zp_mat = torch.empty((H, N), device=V.device, dtype=torch.float32)

            BLOCK_D = min(128, triton.next_power_of_2(max(16, D)))
            grid = (B * H, triton.cdiv(D, BLOCK_D))
            flash_decay_fwd_kernel[grid](
                V_contig,
                lam_f32,
                y_f32,
                z_mat,
                zp_mat,
                V_contig.stride(0),
                V_contig.stride(1),
                V_contig.stride(2),
                V_contig.stride(3),
                y_f32.stride(0),
                y_f32.stride(1),
                y_f32.stride(2),
                y_f32.stride(3),
                B,
                H,
                N,
                D=D,
                BLOCK_D=BLOCK_D,
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
            flash_decay_bwd_kernel[grid](
                g_contig,
                y_saved,
                lam_saved,
                z_mat,
                zp_mat,
                grad_v_f32,
                grad_lam_partial,
                g_contig.stride(0),
                g_contig.stride(1),
                g_contig.stride(2),
                g_contig.stride(3),
                y_saved.stride(0),
                y_saved.stride(1),
                y_saved.stride(2),
                y_saved.stride(3),
                grad_v_f32.stride(0),
                grad_v_f32.stride(1),
                grad_v_f32.stride(2),
                grad_v_f32.stride(3),
                H,
                N,
                D=D,
                NUM_D_BLOCKS=num_d_blocks,
                BLOCK_D=BLOCK_D,
            )
            grad_lam = grad_lam_partial.view(B, H, num_d_blocks).sum(dim=(0, 2))
            return grad_lam.to(ctx.input_lam_dtype), grad_v_f32.to(ctx.input_v_dtype)

        compute_dtype = getattr(ctx, "compute_dtype", torch.float32)
        grad_lam, grad_v = _pytorch_decay_bwd(grad_out, y_saved, lam_saved, z_mat, zp_mat, compute_dtype)
        return grad_lam.to(ctx.input_lam_dtype), grad_v.to(ctx.input_v_dtype)


def triton_global_decay(lam_scores: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Differentiable causal exponential decay operator over sequences.

    Parameters
    ----------
    lam_scores : torch.Tensor
        Decay coefficients per head, shape (H,), values in [0, 1].
    V : torch.Tensor
        Value tensor of shape (B, H, N, D).

    Returns
    -------
    torch.Tensor
        Normalized decayed values of shape (B, H, N, D) with full autograd support
        on both CPU and CUDA (Triton).
    """
    out: torch.Tensor = FlashDecayFunction.apply(lam_scores, V)
    return out
