from __future__ import annotations

import math
import time
from typing import Optional

import torch

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.model_config import KilatConfig
from kilat.engine.checkpoint import capture_rng_state, get_device, restore_rng_state
from kilat.models.attention import KilatAttention, KilatAttentionRoPE
from kilat.models.ffn import DeepSeekMoE
from kilat.models.kernels.flash_decay import (
    HAS_TRITON,
    _reference_global_decay_fp64,
    compute_decay_normalizer_from_logit,
    triton_global_decay,
)
from kilat.models.transformer import KilatTransformer


def _relative_linf_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    diff = float((candidate.double() - reference.double()).abs().max().item())
    denom = max(1.0, float(reference.double().abs().max().item()))
    return diff / denom


def run_gate3_equivalence(
    model_config: Optional[KilatConfig] = None,
    model: Optional[KilatTransformer] = None,
) -> GateResult:
    """
    Gate 3: Implementation Equivalence (`G3_EQUIVALENCE`).
    Verifies:
      1. Fast/Triton Global Decay forward & backward vs exact FP64 causal matrix across FP32/FP16/BF16,
         plus near-1 horizon stability (`log_lambda = 25.0`, lambda -> 1^-).
      2. Full prefill (S=12) vs token-by-token HybridCache decode vs chunked prefill (5+4+3)
         across all 6 combinations of `recall_ratio in {0.0, 0.5, 1.0}` x `use_rope in {False, True}`.
      3. Vectorized token-sorted `DeepSeekMoE` vs `_naive_reference_forward` + uniform aux_loss == alpha_aux.
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    try:
        torch.manual_seed(20261111)

        # ── 1. Global Decay Precision Equivalence vs Exact FP64 Reference ────────
        # Tests both CPU PyTorch fallback and CUDA Triton kernel when hardware is available
        V_ref = torch.randn(2, 3, 10, 8, dtype=torch.float64, requires_grad=True)
        lam_ref = torch.tensor([0.65, 0.92, 0.9995], dtype=torch.float64, requires_grad=True)
        out_ref = _reference_global_decay_fp64(lam_ref, V_ref)
        grad_out_ref = torch.randn_like(out_ref)
        out_ref.backward(grad_out_ref)
        if not (V_ref.grad is not None and lam_ref.grad is not None):
            raise RuntimeError("Assurance invariant violated (survives python -O): V_ref.grad is not None and lam_ref.grad is not None")
        dtype_specs = [
            ("fp32", torch.float32, 1e-5, 5e-5),
            ("fp16", torch.float16, 5e-3, 1e-2),
            ("bf16", torch.bfloat16, 2e-2, 3e-2),
        ]
        target_devices = [torch.device("cpu")]
        if torch.cuda.is_available() and HAS_TRITON:
            target_devices.append(torch.device("cuda:0"))

        decay_errors: dict[str, float] = {}
        for dev in target_devices:
            dev_tag = "cuda_triton" if dev.type == "cuda" else "cpu"
            for label, dt, fwd_tol, bwd_tol in dtype_specs:
                V_dt = V_ref.detach().to(device=dev, dtype=dt).requires_grad_(True)
                lam_dt = lam_ref.detach().to(device=dev, dtype=torch.float32).requires_grad_(True)
                out_dt = triton_global_decay(lam_dt, V_dt)
                if not torch.isfinite(out_dt).all():
                    raise AssuranceError(
                        AssuranceFailureCode.ERR_EQUIVALENCE_DRIFT,
                        f"Non-finite Global Decay forward output in {dev_tag}_{label}.",
                    )
                out_dt.backward(grad_out_ref.to(device=dev, dtype=dt))
                if not (V_dt.grad is not None and lam_dt.grad is not None):
                    raise RuntimeError("Assurance invariant violated (survives python -O): V_dt.grad is not None and lam_dt.grad is not None")
                err_fwd = _relative_linf_error(out_dt.cpu(), out_ref)
                err_dv = _relative_linf_error(V_dt.grad.cpu(), V_ref.grad)
                err_dlam = _relative_linf_error(lam_dt.grad.cpu(), lam_ref.grad)
                decay_errors[f"{dev_tag}_{label}_fwd_rel_err"] = err_fwd
                decay_errors[f"{dev_tag}_{label}_grad_rel_err"] = max(err_dv, err_dlam)

                if err_fwd > fwd_tol or max(err_dv, err_dlam) > bwd_tol:
                    raise AssuranceError(
                        AssuranceFailureCode.ERR_EQUIVALENCE_DRIFT,
                        f"Global Decay {dev_tag}_{label} equivalence exceeded tolerance "
                        f"(fwd={err_fwd:.3e} vs {fwd_tol:.1e}, grad={max(err_dv, err_dlam):.3e} vs {bwd_tol:.1e}).",
                        decay_errors,
                    )

        # Near-unity horizon singularity guard (log_lambda = 25.0 => 1 - float32(lam) == 0.0)
        extreme_logit = torch.tensor([25.0, 30.0], dtype=torch.float32)
        for step_idx in (0, 5, 11):
            z_val = compute_decay_normalizer_from_logit(extreme_logit, step_idx)
            expected_z = float(step_idx + 1)
            if not torch.isfinite(z_val).all() or float((z_val - expected_z).abs().max().item()) > 1e-3:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_DECAY_NORMALIZATION_DIVERGENCE,
                    f"Horizon normalizer diverged at log_lambda>=25, step={step_idx}: got {z_val.tolist()} vs {expected_z}.",
                )

        # Also verify live model attention modules if provided (catches Saboteur C-05 hooks on model)
        if model is not None:
            dev = get_device(model)
            was_train = model.training
            model.eval()
            saved_lambdas: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
            for mod in model.modules():
                log_lam = getattr(mod, "log_lambda", None)
                if isinstance(log_lam, torch.nn.Parameter):
                    saved_lambdas.append((log_lam, log_lam.detach().clone()))
                    with torch.no_grad():
                        log_lam.fill_(25.0)
            try:
                if saved_lambdas:
                    v_sz = int(model.config.vocab_size)
                    toks = torch.randint(0, v_sz, (1, 6), device=dev, dtype=torch.long)
                    with torch.no_grad():
                        logits_full_ext = model(toks, return_dict=True).logits
                        past_ext = None
                        decode_steps = []
                        for s_i in range(6):
                            out_step = model(
                                toks[:, s_i : s_i + 1],
                                past_key_values=past_ext,
                                use_cache=True,
                                return_dict=True,
                            )
                            past_ext = out_step.past_key_values
                            decode_steps.append(out_step.logits)
                        logits_dec_ext = torch.cat(decode_steps, dim=1)
                    if (
                        not torch.isfinite(logits_full_ext).all()
                        or not torch.isfinite(logits_dec_ext).all()
                        or float((logits_full_ext - logits_dec_ext).abs().max().item()) > 1e-4
                    ):
                        raise AssuranceError(
                            AssuranceFailureCode.ERR_DECAY_NORMALIZATION_DIVERGENCE,
                            "Model Global Decay diverged or produced NaN/Inf at extreme horizon log_lambda=25.0.",
                        )
            finally:
                with torch.no_grad():
                    for p_lam, orig_tensor in saved_lambdas:
                        p_lam.copy_(orig_tensor)
                if was_train:
                    model.train()

        # ── 2. Full Prefill vs Token-by-Token & Chunked HybridCache Matrix ───────
        max_cache_diff = 0.0
        S_total = 12
        for use_rope in (False, True):
            for recall_ratio in (0.0, 0.5, 1.0):
                attn_mod: torch.nn.Module
                if use_rope:
                    attn_mod = KilatAttentionRoPE(
                        n_embd=32,
                        n_head=4,
                        recall_ratio=recall_ratio,
                        latent_dim=8,
                        rope_head_dim=8,
                        attn_drop=0.0,
                    ).eval()
                else:
                    attn_mod = KilatAttention(
                        n_embd=32,
                        n_head=4,
                        recall_ratio=recall_ratio,
                        latent_dim=8,
                        attn_drop=0.0,
                    ).eval()

                x_seq = torch.randn(2, S_total, 32, dtype=torch.float32)
                with torch.no_grad():
                    out_prefill = attn_mod(x_seq, use_cache=False)

                    # 1-by-1 token generation
                    cache_step = None
                    step_outs = []
                    for t_idx in range(S_total):
                        o_t, cache_step = attn_mod(x_seq[:, t_idx : t_idx + 1, :], past_key_values=cache_step, use_cache=True)
                        step_outs.append(o_t)
                    out_stepwise = torch.cat(step_outs, dim=1)

                    # Chunked prefill (5 + 4 + 3 = 12)
                    c1, cache_c = attn_mod(x_seq[:, :5, :], past_key_values=None, use_cache=True)
                    c2, cache_c = attn_mod(x_seq[:, 5:9, :], past_key_values=cache_c, use_cache=True)
                    c3, cache_c = attn_mod(x_seq[:, 9:12, :], past_key_values=cache_c, use_cache=True)
                    out_chunked = torch.cat([c1, c2, c3], dim=1)

                diff_step = float((out_prefill - out_stepwise).abs().max().item())
                diff_chunk = float((out_prefill - out_chunked).abs().max().item())
                worst = max(diff_step, diff_chunk)
                max_cache_diff = max(max_cache_diff, worst)
                if worst > 1e-5:
                    raise AssuranceError(
                        AssuranceFailureCode.ERR_CACHE_EQUIVALENCE_DRIFT,
                        f"HybridCache drift {worst:.3e} > 1e-5 for recall_ratio={recall_ratio}, use_rope={use_rope}.",
                        {"recall_ratio": recall_ratio, "use_rope": use_rope, "drift": worst},
                    )

        # ── 3. DeepSeekMoE Sorted Dispatch vs Naive Loop + Uniform Aux Loss ─────
        max_moe_diff = 0.0
        alpha_aux = 0.03
        for k_active in (1, 2, 4):
            moe = DeepSeekMoE(
                dim=16,
                num_routed_experts=4,
                active_experts=k_active,
                num_shared_experts=1,
                dropout=0.0,
                aux_loss_coef=alpha_aux,
                device_balance_coef=0.0,
            ).eval()
            x_m = torch.randn(3, 8, 16, dtype=torch.float32)
            with torch.no_grad():
                out_fast, aux_fast = moe(x_m)
                out_naive, aux_naive = moe._naive_reference_forward(x_m)

            diff_out = float((out_fast - out_naive).abs().max().item())
            diff_aux = float((aux_fast - aux_naive).abs().max().item())
            max_moe_diff = max(max_moe_diff, diff_out, diff_aux)
            if diff_out > 1e-6 or diff_aux > 1e-6:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MOE_EQUIVALENCE_DRIFT,
                    f"DeepSeekMoE sorted dispatch drifted from naive reference (out={diff_out:.3e}, aux={diff_aux:.3e}).",
                )

            # Uniform router weights => exact balance => aux_loss == alpha_aux
            with torch.no_grad():
                moe.router.weight.zero_()
                _u_out, u_aux = moe(x_m)
            if abs(float(u_aux.item()) - alpha_aux) > 1e-6:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MOE_EQUIVALENCE_DRIFT,
                    f"Uniform MoE router aux_loss={float(u_aux.item()):.6f} != expected {alpha_aux:.6f} for K={k_active}.",
                )

        _ = model_config
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="G3_EQUIVALENCE",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                **{k: round(v, 8) for k, v in decay_errors.items()},
                "max_hybrid_cache_diff": max_cache_diff,
                "max_moe_reference_diff": max_moe_diff,
            },
        )
    finally:
        restore_rng_state(saved_rng)
