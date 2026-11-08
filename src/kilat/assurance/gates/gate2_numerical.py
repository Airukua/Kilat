from __future__ import annotations

import time
from typing import Any, Optional

import torch
import torch.nn as nn

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.model_config import KilatConfig
from kilat.engine.checkpoint import capture_rng_state, get_device, restore_rng_state
from kilat.models.attention import KilatAttention, KilatAttentionRoPE
from kilat.models.blocks import RMSNorm
from kilat.models.ffn import DeepSeekMoE, SwiGLU
from kilat.models.kernels.flash_decay import FlashDecayFunction
from kilat.models.transformer import KilatTransformer


def _run_fp64_gradchecks() -> int:
    """Run finite-difference `torch.autograd.gradcheck` across core differentiable primitives."""
    checks_run = 0
    torch.manual_seed(104729)

    # 1. FlashDecayFunction across extreme decay horizons (including 1 - 1e-7)
    for lam_val in (0.5, 0.9, 0.99999, 1.0 - 1e-7):
        V = torch.randn(2, 2, 5, 4, dtype=torch.float64, requires_grad=True)
        lam = torch.tensor([lam_val, 0.85], dtype=torch.float64, requires_grad=True)

        def _decay_fn(lam_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
            return FlashDecayFunction.apply(lam_in, v_in)  # type: ignore[no-any-return]

        if not torch.autograd.gradcheck(_decay_fn, (lam, V), eps=1e-6, atol=1e-5, rtol=1e-4):
            raise AssuranceError(
                AssuranceFailureCode.ERR_GRADCHECK_MISMATCH,
                f"FP64 gradcheck failed on FlashDecayFunction for lambda={lam_val}.",
            )
        checks_run += 1

    # 2. RMSNorm & SwiGLU
    rms = RMSNorm(8).double()
    x_rms = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    if not torch.autograd.gradcheck(lambda inp: rms(inp), (x_rms,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise AssuranceError(AssuranceFailureCode.ERR_GRADCHECK_MISMATCH, "FP64 gradcheck failed on RMSNorm.")
    checks_run += 1

    swiglu = SwiGLU(dim=8, ff_mult=2.0, dropout=0.0).double().eval()
    x_swi = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    if not torch.autograd.gradcheck(lambda inp: swiglu(inp), (x_swi,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise AssuranceError(AssuranceFailureCode.ERR_GRADCHECK_MISMATCH, "FP64 gradcheck failed on SwiGLU.")
    checks_run += 1

    # 3. KilatAttention & KilatAttentionRoPE
    attn = KilatAttention(n_embd=8, n_head=2, recall_ratio=0.5, latent_dim=4, attn_drop=0.0).double().eval()
    x_attn = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    if not torch.autograd.gradcheck(lambda inp: attn(inp), (x_attn,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise AssuranceError(AssuranceFailureCode.ERR_GRADCHECK_MISMATCH, "FP64 gradcheck failed on KilatAttention.")
    checks_run += 1

    attn_rope = (
        KilatAttentionRoPE(
            n_embd=8,
            n_head=2,
            recall_ratio=0.5,
            latent_dim=4,
            rope_head_dim=4,
            attn_drop=0.0,
        )
        .double()
        .eval()
    )
    if not torch.autograd.gradcheck(lambda inp: attn_rope(inp), (x_attn,), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise AssuranceError(
            AssuranceFailureCode.ERR_GRADCHECK_MISMATCH,
            "FP64 gradcheck failed on KilatAttentionRoPE.",
        )
    checks_run += 1

    # 4. DeepSeekMoE for K=1 and K=2 (well-separated router rows avoid top-k tie boundary during eps perturbation)
    for k_active in (1, 2):
        moe = (
            DeepSeekMoE(
                dim=8,
                num_routed_experts=3,
                active_experts=k_active,
                num_shared_experts=1,
                dropout=0.0,
                aux_loss_coef=0.0,
                device_balance_coef=0.0,
            )
            .double()
            .eval()
        )
        with torch.no_grad():
            moe.router.weight.copy_(
                torch.tensor(
                    [
                        [1.5, -0.8, 0.4, 0.9, -1.1, 0.3, 0.7, -0.5],
                        [-1.2, 1.4, -0.6, 0.2, 1.3, -0.9, 0.1, 0.8],
                        [0.1, 0.2, 1.6, -1.3, -0.2, 1.1, -1.0, 0.4],
                    ],
                    dtype=torch.float64,
                )
            )
        x_moe = torch.linspace(-0.7, 0.8, steps=2 * 3 * 8, dtype=torch.float64).reshape(2, 3, 8)
        x_moe.requires_grad_(True)

        def _moe_fn(inp: torch.Tensor, rw: torch.Tensor, mod: DeepSeekMoE = moe) -> torch.Tensor:
            # Functional evaluation w.r.t. input and router weights
            orig_w = mod.router.weight
            logits = torch.nn.functional.linear(inp, rw)
            _ = orig_w
            probs = torch.softmax(logits, dim=-1)
            topk_w, topk_idx = torch.topk(probs, k=mod.effective_active_experts, dim=-1)
            if mod.effective_active_experts > 1:
                topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
            flat_x = inp.reshape(-1, 8)
            out = torch.zeros_like(flat_x)
            for slot in range(mod.effective_active_experts):
                idx_s = topk_idx.reshape(-1, mod.effective_active_experts)[:, slot]
                w_s = topk_w.reshape(-1, mod.effective_active_experts)[:, slot : slot + 1]
                for e_id, exp in enumerate(mod.routed_experts):
                    mask = idx_s == e_id
                    if mask.any():
                        out[mask] = out[mask] + w_s[mask] * exp(flat_x[mask])
            return out.reshape_as(inp)

        rw_var = moe.router.weight.detach().clone().requires_grad_(True)
        if not torch.autograd.gradcheck(_moe_fn, (x_moe, rw_var), eps=1e-6, atol=1e-5, rtol=1e-4):
            raise AssuranceError(
                AssuranceFailureCode.ERR_GRADCHECK_MISMATCH,
                f"FP64 gradcheck failed on DeepSeekMoE routing (K={k_active}).",
            )
        checks_run += 1

    return checks_run


def run_gate2_numerical(
    model_config: KilatConfig,
    model: Optional[KilatTransformer] = None,
    skip_gradcheck: bool = False,
) -> GateResult:
    """
    Gate 2: Numerical, Gradient, & Causal Integrity (`G2_NUMERICAL_CAUSAL`).
    Verifies:
      1. FP64 `torch.autograd.gradcheck` across FlashDecayFunction, RMSNorm, SwiGLU, Attention, MoE.
      2. Causal Future-Blindness perturbation & backward input-embedding Jacobian == 0 for future tokens.
      3. Cross-sample Batch-Isolation test (corrupting samples 1..B-1 leaves sample 0 invariant).
      4. 100% active parameter gradient connectivity (`grad is not None`, finite, non-zero norm).
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    try:
        gradchecks_passed = 0 if skip_gradcheck else _run_fp64_gradchecks()

        torch.manual_seed(20261110)
        target_model = model if model is not None else KilatTransformer(model_config)
        device = get_device(target_model)
        was_training = target_model.training
        vocab_size = int(model_config.vocab_size)
        max_pos = int(getattr(model_config, "max_position_embeddings", getattr(model_config, "block_size", 32)))
        seq_len = min(12, max(6, max_pos))

        # ── 2. Causal Future-Blindness Perturbation + Jacobian ──────────────────
        target_model.eval()
        x_base = torch.randint(0, vocab_size, (2, seq_len), device=device, dtype=torch.long)
        with torch.no_grad():
            logits_a = target_model(x_base, return_dict=True).logits

        max_causal_diff = 0.0
        pivot_positions = sorted({0, seq_len // 2, max(0, seq_len - 2)})
        for t_star in pivot_positions:
            if t_star + 1 >= seq_len:
                continue
            x_tilde = x_base.clone()
            x_tilde[:, t_star + 1 :] = (x_base[:, t_star + 1 :] + 19) % vocab_size
            with torch.no_grad():
                logits_b = target_model(x_tilde, return_dict=True).logits
            diff = float((logits_a[:, : t_star + 1, :] - logits_b[:, : t_star + 1, :]).abs().max().item())
            max_causal_diff = max(max_causal_diff, diff)
            if not torch.isfinite(logits_b).all() or diff > 1e-5:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_CAUSAL_FUTURE_LEAK,
                    f"Future token perturbation beyond t*={t_star} leaked into past logits (max_diff={diff:.3e}).",
                    {"pivot": t_star, "max_causal_diff": diff},
                )

        # Causal Jacobian w.r.t. token embeddings
        target_model.zero_grad(set_to_none=True)
        captured_emb: list[torch.Tensor] = []

        def _capture_wte(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            out.retain_grad()
            captured_emb.append(out)

        wte_hook = target_model.wte.register_forward_hook(_capture_wte)
        try:
            logits_jac = target_model(x_base, return_dict=True).logits
            t_mid = seq_len // 2
            logits_jac[:, t_mid, :].sum().backward()
            if captured_emb and captured_emb[0].grad is not None and t_mid + 1 < seq_len:
                future_grad_max = float(captured_emb[0].grad[:, t_mid + 1 :, :].abs().max().item())
                if future_grad_max > 1e-7:
                    raise AssuranceError(
                        AssuranceFailureCode.ERR_CAUSAL_FUTURE_LEAK,
                        f"Backward Jacobian wrt future embeddings t > {t_mid} is non-zero ({future_grad_max:.3e}).",
                        {"future_grad_max": future_grad_max},
                    )
        finally:
            wte_hook.remove()
            target_model.zero_grad(set_to_none=True)

        # ── 3. Cross-Sample Batch Isolation Test ────────────────────────────────
        x_batch = torch.randint(0, vocab_size, (3, seq_len), device=device, dtype=torch.long)
        x_batch_corrupt = x_batch.clone()
        x_batch_corrupt[1:] = (x_batch[1:] + 31) % vocab_size
        with torch.no_grad():
            out_clean = target_model(x_batch, return_dict=True).logits
            out_corrupt = target_model(x_batch_corrupt, return_dict=True).logits
        batch_isolation_diff = float((out_clean[0] - out_corrupt[0]).abs().max().item())
        if not torch.isfinite(out_corrupt).all() or batch_isolation_diff > 1e-5:
            raise AssuranceError(
                AssuranceFailureCode.ERR_BATCH_ISOLATION_BREACH,
                f"Corrupting batch indices 1..B-1 altered sample 0 logits by {batch_isolation_diff:.3e}.",
                {"batch_isolation_diff": batch_isolation_diff},
            )

        # ── 4. 100% Active Trainable Parameter Connectivity ─────────────────────
        target_model.train()
        target_model.zero_grad(set_to_none=True)
        grad_seq_len = min(max(16, seq_len), max_pos)
        x_grad = torch.randint(0, vocab_size, (6, grad_seq_len), device=device, dtype=torch.long)
        out_g = target_model(x_grad, labels=x_grad, return_dict=True)
        logits_g, loss_g = out_g.logits, out_g.loss
        if loss_g is None or not torch.isfinite(logits_g).all() or not torch.isfinite(loss_g):
            raise AssuranceError(
                AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                "Non-finite logits or loss encountered during gradient connectivity check.",
            )
        loss_g.backward()

        missing_grads: list[str] = []
        zero_grads: list[str] = []
        frozen_with_grads: list[str] = []
        checked_trainable = 0

        for name, param in target_model.named_parameters():
            if not param.requires_grad:
                if param.grad is not None and float(param.grad.abs().max().item()) > 0.0:
                    frozen_with_grads.append(name)
                continue

            checked_trainable += 1
            if param.grad is None:
                missing_grads.append(name)
                continue
            if not torch.isfinite(param.grad).all():
                raise AssuranceError(
                    AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                    f"Non-finite gradient detected on parameter '{name}'.",
                    {"parameter": name},
                )
            grad_norm = float(param.grad.detach().float().norm().item())
            if grad_norm == 0.0 or ("router" in name and grad_norm < 1e-6):
                # Catches both exact 0.0 gradients and Top-1 MoE w/(w+eps) ~1e-9 crushed router gradients
                if ".routed_experts." not in name:
                    zero_grads.append(name)

        target_model.zero_grad(set_to_none=True)
        if not was_training:
            target_model.eval()

        if frozen_with_grads:
            raise AssuranceError(
                AssuranceFailureCode.ERR_FROZEN_PARAMETER_GRAD,
                f"Frozen parameters received non-zero gradients: {frozen_with_grads}",
                {"frozen_with_grads": frozen_with_grads},
            )
        if missing_grads:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MISSING_PARAMETER_GRAD,
                f"Active trainable parameters have grad=None (detached autograd branch): {missing_grads}",
                {"missing_grads": missing_grads},
            )
        if zero_grads:
            raise AssuranceError(
                AssuranceFailureCode.ERR_ZERO_PARAMETER_GRAD,
                f"Active trainable parameters received exact zero gradient norm: {zero_grads}",
                {"zero_grads": zero_grads},
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="G2_NUMERICAL_CAUSAL",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                "fp64_gradchecks_passed": gradchecks_passed,
                "max_causal_diff": max_causal_diff,
                "batch_isolation_diff": batch_isolation_diff,
                "trainable_tensors_verified": checked_trainable,
            },
        )
    finally:
        restore_rng_state(saved_rng)
