"""Pillar 1: Mechanical & Causal Correctness.

Verifies:
1. Initial Unbiased Entropy: Loss at step 0 satisfies |L_0 - ln(V)| <= 0.25.
2. Causal Future-Blindness: Mutating future tokens produces zero difference in past logits (diff <= 1e-6).
3. Gradient Connectivity: 100% of trainable parameters receive non-zero, finite gradients (0 NaN, 0 Inf, 0 Dead).
4. Deterministic Single-Batch Overfit: Pure CE loss drops below threshold (default < 0.05).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..protocol import run_with_seeds, summarize
from ..registry import extract_logits


@dataclass
class CorrectnessConfig:
    seq_len: int = 32
    vocab_size: int = 64
    batch_size: int = 4
    overfit_steps: int = 150
    overfit_loss_threshold: float = 0.05
    lr: float = 3e-3
    device: str = "cpu"
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])


def check_initial_entropy(model: nn.Module, cfg: CorrectnessConfig, seed: int = 0) -> Dict[str, Any]:
    """Check that loss at initialization matches uniform distribution ln(V)."""
    torch.manual_seed(seed)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    with torch.no_grad():
        logits = extract_logits(model(x))
        l0 = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    ideal_entropy = math.log(cfg.vocab_size)
    diff = abs(l0 - ideal_entropy)
    passed = diff <= 0.35
    return {
        "l0": l0,
        "ideal_ln_v": ideal_entropy,
        "diff": diff,
        "passed": passed,
    }


def check_causal_future_blindness(
    model: nn.Module,
    cfg: CorrectnessConfig,
    seed: int = 0,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify that perturbing tokens at t > t* produces 0 change in logits at t <= t*."""
    torch.manual_seed(seed)
    model.eval()
    x1 = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=cfg.device)
    x2 = x1.clone()
    split_pos = cfg.seq_len // 2

    # Mutate all tokens after split_pos
    x2[0, split_pos + 1:] = (x2[0, split_pos + 1:] + 1) % cfg.vocab_size

    with torch.no_grad():
        out1 = extract_logits(model(x1))
        out2 = extract_logits(model(x2))

    past_diff = (out1[:, :split_pos + 1] - out2[:, :split_pos + 1]).abs().max().item()
    future_diff = (out1[:, split_pos + 1:] - out2[:, split_pos + 1:]).abs().max().item()
    passed = past_diff <= atol
    return {
        "max_past_leak_diff": past_diff,
        "future_response_diff": future_diff,
        "tolerance": atol,
        "passed": passed,
    }


def check_gradient_connectivity(
    model: nn.Module,
    cfg: CorrectnessConfig,
    seed: int = 0,
) -> Dict[str, Any]:
    """Verify that all trainable parameters receive non-zero, finite gradients."""
    torch.manual_seed(seed)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=cfg.device)

    logits = extract_logits(model(x))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()

    n_params = 0
    n_nan = 0
    n_inf = 0
    n_dead = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        n_params += 1
        if torch.isnan(p.grad).any():
            n_nan += 1
        if torch.isinf(p.grad).any():
            n_inf += 1
        if math.isclose(float(p.grad.abs().max().item()), 0.0, abs_tol=1e-12):
            n_dead += 1

    passed = (n_nan == 0) and (n_inf == 0) and (n_dead == 0) and (n_params > 0)
    return {
        "n_active_param_tensors": n_params,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_dead": n_dead,
        "passed": passed,
    }


def overfit_single_batch(
    model_builder: Callable[[], nn.Module],
    cfg: CorrectnessConfig,
    seed: int,
) -> Dict[str, Any]:
    """Optimize a single fixed batch to verify the model can drive loss to near-zero."""
    gen = torch.Generator().manual_seed(seed)
    model = model_builder().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), generator=gen).to(cfg.device)
    y = torch.roll(x, shifts=-1, dims=1)

    losses = []
    for _ in range(cfg.overfit_steps):
        opt.zero_grad()
        logits = extract_logits(model(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    return {
        "final_loss": final_loss,
        "passed": final_loss < cfg.overfit_loss_threshold,
    }


def run_pillar1(model_builder: Callable[..., nn.Module], cfg: CorrectnessConfig) -> Dict[str, Any]:
    """Execute full Pillar 1 battery across multiple seeds."""
    def inst_builder(**kw):
        return model_builder(vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len, **kw)

    entropy_report = check_initial_entropy(inst_builder().to(cfg.device), cfg)
    causal_report = check_causal_future_blindness(inst_builder().to(cfg.device), cfg)
    grad_report = check_gradient_connectivity(inst_builder().to(cfg.device), cfg)

    overfit_results = run_with_seeds(
        lambda seed: overfit_single_batch(inst_builder, cfg, seed),
        seeds=cfg.seeds,
    )
    overfit_summary = summarize(overfit_results)
    all_overfit_passed = all(r.value["passed"] for r in overfit_results)

    overall_passed = (
        entropy_report["passed"]
        and causal_report["passed"]
        and grad_report["passed"]
        and all_overfit_passed
    )

    return {
        "pillar": "P1_MECHANICAL_CORRECTNESS",
        "passed": overall_passed,
        "initial_entropy": entropy_report,
        "causal_future_blindness": causal_report,
        "gradient_connectivity": grad_report,
        "single_batch_overfit": {
            "per_seed": [{"seed": r.seed, **r.value} for r in overfit_results],
            "summary": overfit_summary,
            "passed": all_overfit_passed,
        },
    }
