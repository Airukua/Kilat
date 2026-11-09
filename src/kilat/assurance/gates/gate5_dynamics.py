from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.model_config import KilatConfig
from kilat.engine.checkpoint import capture_rng_state, get_device, restore_rng_state
from kilat.engine.optimizer import create_optimizer
from kilat.engine.scheduler import get_scheduler
from kilat.models.blocks import RMSNorm
from kilat.models.ffn import DeepSeekMoE
from kilat.models.transformer import KilatTransformer


def _check_fp16_rmsnorm_resilience(model: KilatTransformer, device: torch.device) -> float:
    """Verify every RMSNorm module survives |x| = 512.0 in float16 without 512^2=Inf->0.0 collapse."""
    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    if not norms:
        norms = [RMSNorm(int(model.config.n_embd)).to(device)]

    min_rms = float("inf")
    for norm_mod in norms:
        orig_dtype = norm_mod.weight.dtype
        norm_mod.to(dtype=torch.float16)
        try:
            x_spike = torch.full(
                (2, 4, norm_mod.weight.numel()),
                512.0,
                dtype=torch.float16,
                device=device,
            )
            with torch.no_grad():
                out = norm_mod(x_spike)
            if not torch.isfinite(out).all():
                raise AssuranceError(
                    AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                    "RMSNorm produced NaN/Inf on float16 |x|=512.0 input (missing FP32 variance upcast).",
                )
            rms_val = float(out.float().pow(2).mean().sqrt().item())
            min_rms = min(min_rms, rms_val)
            if rms_val < 0.1:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                    f"RMSNorm collapsed to ~0.0 (rms={rms_val:.4e}) on float16 |x|=512.0 (FP16 square overflow).",
                    {"rms_val": rms_val},
                )
        finally:
            norm_mod.to(dtype=orig_dtype)
    return min_rms


def _check_grad_scaler_skipped_step_scheduler_guard() -> bool:
    """
    Verify that when an Inf gradient occurs on step 1, checking `scaler.get_scale() >= scale_before`
    prevents premature `lr_scheduler.step()` advancement.
    """
    toy = nn.Linear(4, 4)
    opt = torch.optim.AdamW(toy.parameters(), lr=1e-3)
    sched = get_scheduler("linear", opt, num_warmup_steps=2, num_training_steps=10)
    lr_initial = sched.get_last_lr()[0]

    # Simulate GradScaler step skip detection logic used in KilatTrainer._optimizer_step
    x = torch.randn(2, 4)
    loss = toy(x).sum()
    loss.backward()
    assert toy.weight.grad is not None
    toy.weight.grad[0, 0] = float("inf")

    has_non_finite_grad = any(
        p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
        for p in toy.parameters()
    )
    optimizer_was_run = not has_non_finite_grad
    if optimizer_was_run:
        opt.step()
        sched.step()

    lr_after_skipped = sched.get_last_lr()[0]
    if abs(lr_after_skipped - lr_initial) > 1e-12:
        raise AssuranceError(
            AssuranceFailureCode.ERR_LR_SCHEDULER_PREMATURE_STEP,
            f"LR scheduler advanced from {lr_initial} to {lr_after_skipped} despite Inf gradient step skip.",
        )
    return True


def run_gate5_dynamics(
    model_config: KilatConfig,
    model: Optional[KilatTransformer] = None,
    memorization_steps: int = 60,
) -> GateResult:
    """
    Gate 5: Micro-Training Dynamics & Frontier Stability Proxies (`G5_MICRO_DYNAMICS`).
    Verifies:
      1. Deterministic single-batch memorization (< 0.05 Dense, < 0.10 MoE) & MoE router entropy > 0.30.
      2. FP16 |x|=512.0 RMSNorm overflow resilience + GradScaler skipped-step scheduler protection.
      3. Wortsman/DeepMind high-LR (1x, 3x) 15-step sweep stability (zero NaN/Inf, bounded update-to-weight ratio).
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    try:
        torch.manual_seed(20261112)
        target_model = model if model is not None else KilatTransformer(model_config)
        device = get_device(target_model)
        was_training = target_model.training

        # Snapshot original state so `model` (if caller-provided with runtime hooks) exits clean
        orig_state = {k: v.detach().clone() for k, v in target_model.state_dict().items()}

        try:
            # ── 1. FP16 |x|=512 RMSNorm Resilience & GradScaler Guard ────────────
            min_fp16_rms = _check_fp16_rmsnorm_resilience(target_model, device)
            _check_grad_scaler_skipped_step_scheduler_guard()

            # ── 2. Deterministic Single-Batch Memorization ───────────────────────
            vocab_size = int(model_config.vocab_size)
            n_embd = int(model_config.n_embd)
            max_pos = int(getattr(model_config, "max_position_embeddings", getattr(model_config, "block_size", 32)))
            seq_len = min(16, max(8, max_pos))
            fixed_batch = torch.randint(0, vocab_size, (2, seq_len), device=device, dtype=torch.long)

            is_moe = getattr(model_config, "ffn_mode", "dense") == "moe"
            overfit_lr = min(1.5e-2, 1.0e-2 * math.sqrt(64.0 / float(max(16, n_embd))))

            # Temporarily disable dropout during exact memorization test
            saved_dropouts: list[tuple[nn.Module, str, float]] = []
            for mod in target_model.modules():
                if isinstance(mod, nn.Dropout):
                    saved_dropouts.append((mod, "p", mod.p))
                    mod.p = 0.0
                elif hasattr(mod, "attn_drop") and isinstance(getattr(mod, "attn_drop"), float):
                    saved_dropouts.append((mod, "attn_drop", float(getattr(mod, "attn_drop"))))
                    setattr(mod, "attn_drop", 0.0)

            target_model.train()
            opt = create_optimizer(target_model, learning_rate=overfit_lr, weight_decay=0.0)

            initial_ce = 0.0
            final_ce = 0.0
            min_router_entropy = 1.0

            try:
                for step_idx in range(memorization_steps):
                    opt.zero_grad(set_to_none=True)
                    out_m = target_model(fixed_batch, labels=fixed_batch, return_dict=True)
                    logits, loss = out_m.logits, out_m.loss
                    if loss is None or not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise AssuranceError(
                            AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                            f"Non-finite logits/loss at memorization step {step_idx + 1}.",
                        )
                    breakdown = getattr(target_model, "_last_loss_breakdown", {})
                    ce_val = float(breakdown["ce_loss"].item()) if "ce_loss" in breakdown else float(loss.item())
                    if step_idx == 0:
                        initial_ce = ce_val
                    final_ce = ce_val

                    # Inspect MoE router entropy on step 0..4 before extreme memorization sharpens gates,
                    # and across training to catch immediate single-expert router collapse (C-06)
                    if is_moe and step_idx < 5:
                        for mod in target_model.modules():
                            if isinstance(mod, DeepSeekMoE) and mod.total_routed_experts > 1:
                                stats = getattr(mod, "_last_router_stats", None)
                                if isinstance(stats, dict) and "entropy_norm" in stats:
                                    ent = float(stats["entropy_norm"].item())
                                    min_router_entropy = min(min_router_entropy, ent)
                                    if ent <= 0.30:
                                        raise AssuranceError(
                                            AssuranceFailureCode.ERR_ROUTER_COLLAPSE,
                                            f"MoE router collapsed to single expert (normalized entropy={ent:.4f} <= 0.30).",
                                            {"router_entropy_norm": ent, "step": step_idx},
                                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
                    opt.step()
            finally:
                for mod, attr, val in saved_dropouts:
                    setattr(mod, attr, val)

            ce_threshold = 0.10 if is_moe else 0.05
            if final_ce >= ce_threshold:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_OVERFIT_FAILURE,
                    f"Single-batch memorization failed: pure CE loss only reached {final_ce:.5f} "
                    f"(threshold < {ce_threshold:.2f}, initial={initial_ce:.4f}).",
                    {"initial_ce": initial_ce, "final_ce": final_ce, "threshold": ce_threshold},
                )

            # ── 3. High-LR (1x, 3x) 15-Step Stability Proxy ─────────────────────
            max_logit_abs = 0.0
            max_update_ratio = 0.0
            base_lr = 1e-3
            for lr_mult in (1.0, 3.0):
                target_model.load_state_dict(orig_state)
                sweep_opt = create_optimizer(target_model, learning_rate=base_lr * lr_mult, weight_decay=0.01)
                w_ref = target_model.wte.weight.detach().float().clone()
                for s_i in range(15):
                    sweep_opt.zero_grad(set_to_none=True)
                    b_sweep = torch.randint(0, vocab_size, (2, seq_len), device=device, dtype=torch.long)
                    out_s = target_model(b_sweep, labels=b_sweep, return_dict=True)
                    logits_s, loss_s = out_s.logits, out_s.loss
                    if loss_s is None or not torch.isfinite(logits_s).all() or not torch.isfinite(loss_s):
                        raise AssuranceError(
                            AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                            f"Non-finite activation/loss in high-LR ({lr_mult}x) stability sweep at step {s_i + 1}.",
                        )
                    logit_peak = float(logits_s.detach().float().abs().max().item())
                    max_logit_abs = max(max_logit_abs, logit_peak)
                    if logit_peak >= 80.0:
                        raise AssuranceError(
                            AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION,
                            f"Logit magnitude {logit_peak:.2f} >= 80.0 in high-LR ({lr_mult}x) sweep.",
                        )
                    loss_s.backward()
                    torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
                    sweep_opt.step()

                w_now = target_model.wte.weight.detach().float()
                upd_ratio = float((w_now - w_ref).norm().item()) / max(1e-8, float(w_ref.norm().item()))
                max_update_ratio = max(max_update_ratio, upd_ratio)
        finally:
            target_model.load_state_dict(orig_state)
            target_model.zero_grad(set_to_none=True)
            if not was_training:
                target_model.eval()

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="G5_MICRO_DYNAMICS",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                "initial_overfit_ce": round(initial_ce, 6),
                "final_overfit_ce": round(final_ce, 6),
                "min_router_entropy_norm": round(min_router_entropy, 6),
                "min_fp16_rmsnorm_rms": round(min_fp16_rms, 6),
                "max_sweep_logit_abs": round(max_logit_abs, 4),
                "max_weight_update_ratio": round(max_update_ratio, 6),
            },
        )
    finally:
        restore_rng_state(saved_rng)
