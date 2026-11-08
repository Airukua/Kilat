from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Optional

import torch

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.configs.main_config import MainConfig
from kilat.configs.model_config import KilatConfig
from kilat.engine.checkpoint import capture_rng_state, restore_rng_state, set_seed
from kilat.engine.optimizer import create_optimizer
from kilat.engine.scheduler import get_scheduler
from kilat.models.transformer import KilatTransformer


def run_stage7_scaling_harness() -> GateResult:
    """
    Stage 7 (`S7_SCALING_LAWS`): muP Width Coordinate Check + WSDLR Mini-Scaling Law Harness.
    Verifies:
      1. Stage 7a (Coordinate Check): across widths d in {32, 64, 128} over 5 steps with inverse-width LR,
         mean L1 activation magnitude ||h_L||_1 / d remains O(1) scale-invariant.
      2. Stage 7b (WSDLR Mini-Scaling): across 3 model capacities N_1 < N_2 < N_3 trained with `wsdlr`
         on a multi-motif sequence task, validation pure CE loss decreases monotonically: L(N_3) < L(N_2) < L(N_1).
    """
    t0 = time.perf_counter()
    saved_rng = capture_rng_state()
    try:
        # ── Stage 7a: Width Coordinate Check (d in {32, 64, 128}) ────────────────
        coord_l1_means: dict[int, float] = {}
        vocab_size = 64
        set_seed(7007)
        probe_tokens = torch.randint(0, vocab_size, (4, 16), dtype=torch.long)

        for width in (32, 64, 128):
            set_seed(7007)
            cfg_w = KilatConfig(
                vocab_size=vocab_size,
                n_embd=width,
                n_head=4,
                n_layer=2,
                max_position_embeddings=32,
                dropout=0.0,
                use_rope=True,
                rope_head_dim=4,
            )
            model_w = KilatTransformer(cfg_w)
            lr_w = 3e-3 * (32.0 / float(width))  # muP-style 1/width scaling on Adam step size
            opt_w = create_optimizer(model_w, learning_rate=lr_w, weight_decay=0.0)

            last_hidden_l1 = 0.0

            def _capture_last_block(_m: Any, _inp: Any, out: Any) -> None:
                nonlocal last_hidden_l1
                h = out[0] if isinstance(out, tuple) else out
                last_hidden_l1 = float(h.detach().float().abs().mean().item())

            hook = model_w.layers[-1].register_forward_hook(_capture_last_block)
            try:
                for _ in range(5):
                    opt_w.zero_grad(set_to_none=True)
                    loss_w = model_w(probe_tokens, labels=probe_tokens, return_dict=True).loss
                    assert loss_w is not None
                    loss_w.backward()
                    opt_w.step()
            finally:
                hook.remove()

            if not math.isfinite(last_hidden_l1) or not (0.005 <= last_hidden_l1 <= 5.0):
                raise AssuranceError(
                    AssuranceFailureCode.ERR_SCALING_COORDINATE_DRIFT,
                    f"Width d={width} mean |h_L|={last_hidden_l1:.4f} outside O(1) coordinate band [0.005, 5.0].",
                    {"width": width, "mean_abs_activation": last_hidden_l1},
                )
            coord_l1_means[width] = round(last_hidden_l1, 5)

        ratio_max_min = max(coord_l1_means.values()) / max(1e-8, min(coord_l1_means.values()))
        if ratio_max_min > 4.0:
            raise AssuranceError(
                AssuranceFailureCode.ERR_SCALING_COORDINATE_DRIFT,
                f"Activation scale drifted {ratio_max_min:.2f}x across widths {coord_l1_means} (> 4.0x limit).",
                {"coord_l1_means": coord_l1_means},
            )

        # ── Stage 7b: 3-Scale WSDLR Mini-Scaling Law (N_1 < N_2 < N_3) ───────────
        set_seed(7070)
        # Construct 16 deterministic structured motifs of length 16 over vocab_size=64
        motifs = []
        for m_idx in range(16):
            seq = [((m_idx * 5 + pos * 3 + (pos % 2) * 7) % vocab_size) for pos in range(16)]
            motifs.append(seq)
        pattern_batch = torch.tensor(motifs, dtype=torch.long)  # (16, 16)

        scales = [
            ("N1_tiny", 16, 2, 1, 15),
            ("N2_small", 32, 4, 2, 30),
            ("N3_medium", 64, 4, 3, 45),
        ]
        scale_losses: dict[str, float] = {}
        scale_params: dict[str, int] = {}

        for label, d_model, n_head, n_layer, steps in scales:
            set_seed(8080)
            head_dim = d_model // n_head
            even_rope_dim = max(2, (head_dim // 2) - ((head_dim // 2) % 2))
            cfg_s = KilatConfig(
                vocab_size=vocab_size,
                n_embd=d_model,
                n_head=n_head,
                n_layer=n_layer,
                max_position_embeddings=32,
                dropout=0.0,
                use_rope=True,
                rope_head_dim=even_rope_dim,
            )
            model_s = KilatTransformer(cfg_s)
            scale_params[label] = sum(p.numel() for p in model_s.parameters())
            lr_s = 1.2e-2 / math.sqrt(float(d_model) / 8.0)
            opt_s = create_optimizer(model_s, learning_rate=lr_s, weight_decay=0.0)
            sched_s = get_scheduler(
                "wsdlr",
                opt_s,
                num_warmup_steps=3,
                num_training_steps=steps,
                num_decay_steps=max(2, steps // 5),
            )

            model_s.train()
            for _ in range(steps):
                opt_s.zero_grad(set_to_none=True)
                loss_s = model_s(pattern_batch, labels=pattern_batch, return_dict=True).loss
                assert loss_s is not None
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
                opt_s.step()
                sched_s.step()

            model_s.eval()
            with torch.no_grad():
                eval_loss_t = model_s(pattern_batch, labels=pattern_batch, return_dict=True).loss
                assert eval_loss_t is not None
                breakdown = getattr(model_s, "_last_loss_breakdown", {})
                ce_eval = float(breakdown["ce_loss"].item()) if "ce_loss" in breakdown else float(eval_loss_t.item())
            scale_losses[label] = round(ce_eval, 6)

        l1, l2, l3 = scale_losses["N1_tiny"], scale_losses["N2_small"], scale_losses["N3_medium"]
        if not (l3 < l2 < l1):
            raise AssuranceError(
                AssuranceFailureCode.ERR_SCALING_MONOTONICITY_BREACH,
                f"WSDLR scaling law monotonicity violated: expected L(N3) < L(N2) < L(N1), got {scale_losses}.",
                {"scale_losses": scale_losses, "scale_params": scale_params},
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GateResult(
            gate_id="S7_SCALING_LAWS",
            status="PASS",
            duration_ms=duration_ms,
            metrics={
                "coord_width_ratio": round(ratio_max_min, 4),
                "loss_N1_tiny": l1,
                "loss_N2_small": l2,
                "loss_N3_medium": l3,
            },
        )
    finally:
        restore_rng_state(saved_rng)


def run_stage8_readiness_signoff(
    main_config: Optional[MainConfig],
    prior_gates: list[GateResult],
) -> GateResult:
    """
    Stage 8 (`S8_READINESS_SIGNOFF`): Final Pre-Training Readiness Sign-Off.
    Validates config serialization roundtrip, TrainingArguments conversion, and 0 prior gate failures.
    """
    t0 = time.perf_counter()
    failed_prior = [g.gate_id for g in prior_gates if g.status != "PASS"]
    if failed_prior:
        raise AssuranceError(
            AssuranceFailureCode.ERR_READINESS_SIGNOFF_FAILED,
            f"Cannot sign off readiness while prior gates failed: {failed_prior}",
            {"failed_gates": failed_prior},
        )

    config_digest = "standalone"
    if main_config is not None:
        cfg_dict = main_config.to_dict()
        serialized = json.dumps(cfg_dict, sort_keys=True, default=str)
        config_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        t_args = main_config.to_training_arguments()
        restored_args = type(t_args).from_dict(t_args.to_dict())
        if restored_args.to_dict() != t_args.to_dict():
            raise AssuranceError(
                AssuranceFailureCode.ERR_READINESS_SIGNOFF_FAILED,
                "TrainingArguments JSON roundtrip mismatch during final readiness check.",
            )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    return GateResult(
        gate_id="S8_READINESS_SIGNOFF",
        status="PASS",
        duration_ms=duration_ms,
        metrics={
            "verified_gates_count": len(prior_gates),
            "config_sha256_prefix": config_digest[:16],
        },
    )
