from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
import torch.nn as nn

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.model_config import KilatConfig
from kilat.engine.checkpoint import capture_rng_state, get_device, restore_rng_state
from kilat.engine.optimizer import create_optimizer
from kilat.models.blocks import RMSNorm
from kilat.models.transformer import KilatTransformer


def _estimate_active_parameters(model: KilatTransformer) -> tuple[int, int]:
    """Return `(total_params, active_params)` accounting for routed MoE sparsity."""
    total_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    if getattr(cfg, "ffn_mode", "dense") != "moe":
        return total_params, total_params

    inactive_routed = 0
    for mod in model.modules():
        routed = getattr(mod, "routed_experts", None)
        k_eff = getattr(mod, "effective_active_experts", None)
        if isinstance(routed, nn.ModuleList) and len(routed) > 0 and isinstance(k_eff, int):
            per_expert = sum(p.numel() for p in routed[0].parameters())
            inactive_count = max(0, len(routed) - k_eff)
            inactive_routed += inactive_count * per_expert

    active_params = max(1, total_params - inactive_routed)
    return total_params, active_params


def run_gate1_theory(
    model_config: KilatConfig,
    model: Optional[KilatTransformer] = None,
    weight_decay: float = 0.1,
) -> GateResult:
    """
    Gate 1: Theoretical & Initialization Sanity (`G1_THEORY_INIT`).
    Verifies:
      1. Unbiased initial pure CE loss |L_CE(0) - ln(V)| <= 0.15 on uniform random tokens.
      2. Residual stream variance growth ratio std(h_L) / std(h_1) in [0.3, 3.0].
      3. Zero 1D / RMSNorm / LayerNorm / bias / log_lambda parameters leaked into weight_decay > 0.
      4. Active vs total parameter accounting and FLOPs/VRAM estimates.
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    try:
        torch.manual_seed(20261109)
        target_model = model if model is not None else KilatTransformer(model_config)
        device = get_device(target_model)
        was_training = target_model.training
        target_model.eval()

        vocab_size = int(model_config.vocab_size)
        max_pos = int(getattr(model_config, "max_position_embeddings", getattr(model_config, "block_size", 32)))
        seq_len = min(32, max(8, max_pos))
        batch_size = 8

        layer_stds: dict[int, float] = {}
        hooks = []
        blocks = target_model.layers
        n_layers = len(blocks)

        def _make_std_hook(idx: int) -> Any:
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                tensor_out = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor_out, torch.Tensor):
                    layer_stds[idx] = float(tensor_out.detach().float().std().item())

            return _hook

        if n_layers >= 1:
            hooks.append(blocks[0].register_forward_hook(_make_std_hook(0)))
        if n_layers >= 2:
            hooks.append(blocks[-1].register_forward_hook(_make_std_hook(n_layers - 1)))

        try:
            with torch.no_grad():
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
                out = target_model(input_ids, labels=input_ids, return_dict=True)
                loss = out.loss
                breakdown = getattr(target_model, "_last_loss_breakdown", None)
                if isinstance(breakdown, dict) and "ce_loss" in breakdown:
                    initial_ce_loss = float(breakdown["ce_loss"].item())
                else:
                    initial_ce_loss = float(loss.item()) if loss is not None else float("nan")
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                target_model.train()

        expected_ln_vocab = math.log(vocab_size)
        ce_diff = abs(initial_ce_loss - expected_ln_vocab)
        if not math.isfinite(initial_ce_loss) or ce_diff > 0.15:
            raise AssuranceError(
                AssuranceFailureCode.ERR_UNBIASED_INIT_LOSS,
                f"Initial pure CE loss {initial_ce_loss:.5f} deviates from ln(V)={expected_ln_vocab:.5f} "
                f"by {ce_diff:.5f} (> 0.15 tolerance).",
                {"initial_ce_loss": initial_ce_loss, "expected_ln_vocab": expected_ln_vocab},
            )

        if n_layers >= 2:
            std_first = max(1e-12, layer_stds.get(0, 1.0))
            std_last = layer_stds.get(n_layers - 1, std_first)
            residual_std_ratio = std_last / std_first
        else:
            residual_std_ratio = 1.0

        if not math.isfinite(residual_std_ratio) or not (0.3 <= residual_std_ratio <= 3.0):
            raise AssuranceError(
                AssuranceFailureCode.ERR_RESIDUAL_VARIANCE_EXPLOSION,
                f"Residual stream std ratio sigma_L/sigma_1 = {residual_std_ratio:.4f} outside [0.3, 3.0].",
                {"residual_std_ratio_L_over_1": residual_std_ratio},
            )

        # Audit weight-decay parameter partitioning (M-03)
        norm_param_ids: set[int] = set()
        for mod in target_model.modules():
            if isinstance(mod, (nn.LayerNorm, RMSNorm)):
                for p in mod.parameters(recurse=False):
                    norm_param_ids.add(id(p))

        param_to_name = {id(p): name for name, p in target_model.named_parameters()}
        effective_wd = weight_decay if weight_decay > 0.0 else 0.1
        optimizer = create_optimizer(target_model, learning_rate=1e-4, weight_decay=effective_wd)

        forbidden_tokens = ("bias", "layernorm", "norm", "rms", "ln_", "log_lambda")
        leaked_names: list[str] = []
        for group in optimizer.param_groups:
            if float(group.get("weight_decay", 0.0)) > 0.0:
                for p in group["params"]:
                    pname = param_to_name.get(id(p), "<unnamed>")
                    pname_low = pname.lower()
                    if (
                        p.ndim < 2
                        or id(p) in norm_param_ids
                        or any(tok in pname_low for tok in forbidden_tokens)
                    ):
                        leaked_names.append(pname)

        if leaked_names:
            raise AssuranceError(
                AssuranceFailureCode.ERR_WEIGHT_DECAY_LEAK,
                f"Found {len(leaked_names)} 1D/norm/log_lambda parameters in weight_decay > 0 group: {leaked_names}",
                {"leaked_params": leaked_names},
            )

        total_params, active_params = _estimate_active_parameters(target_model)
        flops_per_token = 6 * active_params
        # Static FP32 model + AdamW moments (~16 bytes/param) in MB
        estimated_static_vram_mb = (total_params * 16.0) / (1024.0 * 1024.0)

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="G1_THEORY_INIT",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                "initial_ce_loss": round(initial_ce_loss, 6),
                "expected_ln_vocab": round(expected_ln_vocab, 6),
                "residual_std_ratio_L_over_1": round(residual_std_ratio, 6),
                "leaked_decay_1d_params": len(leaked_names),
                "total_params": total_params,
                "active_params": active_params,
                "estimated_flops_per_token": flops_per_token,
                "estimated_static_vram_mb": round(estimated_static_vram_mb, 3),
            },
        )
    finally:
        restore_rng_state(saved_rng)
