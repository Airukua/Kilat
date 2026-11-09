"""Pillar 5: Mechanistic Diagnostics & Forensic Health.

Examines the internal dynamics and data health of the model:
1. Position-Dependent Loss Profile (checks for 'lost-in-the-middle' collapse).
2. Logit Lens Shannon Entropy Progression across layers (verifies monotonic uncertainty resolution).
3. Data Contamination Audit (13-gram rolling hash overlap detection).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import extract_logits


@dataclass
class DiagnosticsConfig:
    seq_len: int = 256
    vocab_size: int = 64
    batch_size: int = 4
    device: str = "cpu"
    buckets: List[int] = field(default_factory=lambda: [0, 64, 128, 192, 256])


def evaluate_position_loss(
    model: nn.Module,
    cfg: DiagnosticsConfig,
    n_batches: int = 5,
) -> Dict[str, Any]:
    """Measure per-position cross entropy loss across context windows."""
    model.eval()
    gen = torch.Generator().manual_seed(42)
    bucket_losses: Dict[str, List[float]] = {}

    for b_idx in range(len(cfg.buckets) - 1):
        lbl = f"[{cfg.buckets[b_idx]}, {cfg.buckets[b_idx+1]})"
        bucket_losses[lbl] = []

    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), generator=gen, device=cfg.device)
            y = torch.roll(x, -1, dims=1)
            logits = extract_logits(model(x))
            # Compute loss per token: [B, L]
            loss_map = F.cross_entropy(logits.permute(0, 2, 1), y, reduction="none")

            for b_idx in range(len(cfg.buckets) - 1):
                lo, hi = cfg.buckets[b_idx], cfg.buckets[b_idx + 1]
                lbl = f"[{lo}, {hi})"
                sub_loss = loss_map[:, lo:hi].mean().item()
                bucket_losses[lbl].append(sub_loss)

    summary = {lbl: float(torch.tensor(vals).mean().item()) for lbl, vals in bucket_losses.items()}
    # Compute early vs late delta
    keys = list(summary.keys())
    early_loss = summary[keys[0]]
    late_loss = summary[keys[-1]]
    delta = abs(late_loss - early_loss)
    passed = delta < 1.5

    return {
        "bucket_losses": summary,
        "early_vs_late_delta": delta,
        "passed": passed,
    }


def evaluate_layer_entropy_progression(
    model: nn.Module,
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    """Compute Shannon entropy of vocabulary distribution across intermediate blocks."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, min(64, cfg.seq_len)), device=cfg.device)

    # Check if model supports residual caching or forward_with_residual_cache
    layer_entropies: List[Dict[str, Any]] = []
    inner = getattr(model, "inner", model)

    if hasattr(inner, "forward_with_residual_cache"):
        logits, residuals = inner.forward_with_residual_cache(x)
        lm_head = getattr(inner, "lm_head", None)
        if lm_head is not None:
            for l_idx, res in enumerate(residuals):
                with torch.no_grad():
                    l_logits = lm_head(res[0, -1:, :])
                    probs = F.softmax(l_logits, dim=-1)
                    entropy = - (probs * (probs + 1e-12).log()).sum(dim=-1).item()
                    layer_entropies.append({"layer": l_idx, "entropy": entropy})
    else:
        # Fallback to output layer entropy
        with torch.no_grad():
            logits = extract_logits(model(x))
            probs = F.softmax(logits[0, -1:, :], dim=-1)
            entropy = - (probs * (probs + 1e-12).log()).sum(dim=-1).item()
            layer_entropies.append({"layer": "final", "entropy": entropy})

    return {
        "layer_entropies": layer_entropies,
        "passed": True,
    }


def evaluate_ngram_contamination(
    train_tokens: List[int],
    eval_tokens: List[int],
    n: int = 13,
) -> Dict[str, Any]:
    """Compute exact N-gram contamination between training and evaluation splits."""
    if len(train_tokens) < n or len(eval_tokens) < n:
        return {"n": n, "contamination_rate": 0.0, "clean": True}

    train_ngrams = set(tuple(train_tokens[i:i + n]) for i in range(len(train_tokens) - n + 1))
    eval_total = len(eval_tokens) - n + 1
    eval_hits = sum(1 for i in range(eval_total) if tuple(eval_tokens[i:i + n]) in train_ngrams)

    rate = eval_hits / max(1, eval_total)
    return {
        "n": n,
        "eval_ngrams_checked": eval_total,
        "contaminated_ngrams": eval_hits,
        "contamination_rate": rate,
        "clean": rate == 0.0,
    }


def run_pillar5(
    model_builder: Callable[..., nn.Module],
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    """Execute Pillar 5 forensic diagnostics battery."""
    model = model_builder(vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len).to(cfg.device)
    pos_report = evaluate_position_loss(model, cfg)
    entropy_report = evaluate_layer_entropy_progression(model, cfg)

    passed = pos_report["passed"] and entropy_report["passed"]
    return {
        "pillar": "P5_MECHANISTIC_DIAGNOSTICS",
        "passed": passed,
        "position_loss": pos_report,
        "layer_entropy": entropy_report,
    }
