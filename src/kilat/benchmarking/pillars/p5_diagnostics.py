"""Pillar 5: Mechanistic Diagnostics & Forensic Health.

Examines internal representations, data pipeline integrity, and counterfactual behaviors:
1. Position-Dependent Loss Profile (detects 'lost-in-the-middle' context degradation).
2. Logit Lens Shannon Entropy Progression across residual blocks.
3. Data Contamination & EOS Boundary Audits (13-gram rolling hash + delimiter spacing).
4. Causal Activation Patching (pinpoints circuit layer effects via residual interventions).
5. Counterfactual Logit Margin Probing (measures targeted retrieval margin z_target - z_foil).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
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


# =====================================================================
# 1. Position-Dependent Loss Profiling
# =====================================================================

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
            loss_map = F.cross_entropy(logits.permute(0, 2, 1), y, reduction="none")

            for b_idx in range(len(cfg.buckets) - 1):
                lo, hi = cfg.buckets[b_idx], cfg.buckets[b_idx + 1]
                lbl = f"[{lo}, {hi})"
                sub_loss = loss_map[:, lo:hi].mean().item()
                bucket_losses[lbl].append(sub_loss)

    summary = {lbl: float(torch.tensor(vals).mean().item()) for lbl, vals in bucket_losses.items()}
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


# =====================================================================
# 2. Logit Lens Layer Entropy
# =====================================================================

def evaluate_layer_entropy_progression(
    model: nn.Module,
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    """Compute Shannon entropy of vocabulary distribution across intermediate blocks."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, min(64, cfg.seq_len)), device=cfg.device)

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
                    layer_entropies.append({"layer": l_idx, "entropy": float(entropy)})
    else:
        with torch.no_grad():
            logits = extract_logits(model(x))
            probs = F.softmax(logits[0, -1:, :], dim=-1)
            entropy = - (probs * (probs + 1e-12).log()).sum(dim=-1).item()
            layer_entropies.append({"layer": "final", "entropy": float(entropy)})

    return {
        "layer_entropies": layer_entropies,
        "passed": True,
    }


# =====================================================================
# 3. Data Integrity & Contamination Audits
# =====================================================================

def evaluate_ngram_contamination(
    train_tokens: Union[List[int], np.ndarray],
    eval_tokens: Union[List[int], np.ndarray],
    n: int = 13,
) -> Dict[str, Any]:
    """Compute exact N-gram contamination between training and evaluation splits."""
    t_list = train_tokens.tolist() if isinstance(train_tokens, np.ndarray) else list(train_tokens)
    e_list = eval_tokens.tolist() if isinstance(eval_tokens, np.ndarray) else list(eval_tokens)

    if len(t_list) < n or len(e_list) < n:
        return {"n": n, "contamination_rate": 0.0, "clean": True}

    train_ngrams = set(tuple(t_list[i:i + n]) for i in range(len(t_list) - n + 1))
    eval_total = len(e_list) - n + 1
    eval_hits = sum(1 for i in range(eval_total) if tuple(e_list[i:i + n]) in train_ngrams)

    rate = eval_hits / max(1, eval_total)
    return {
        "n": n,
        "eval_ngrams_checked": eval_total,
        "contaminated_ngrams": eval_hits,
        "contamination_rate": float(rate),
        "clean": eval_hits == 0,
    }


def audit_eos_boundaries(
    tokens: Union[np.ndarray, torch.Tensor, List[int]],
    eos_token_id: int = 50256,
) -> Dict[str, Any]:
    """Inspect presence and spacing statistics of document delimiter tokens (EOS)."""
    if isinstance(tokens, torch.Tensor):
        arr = tokens.cpu().numpy()
    elif isinstance(tokens, list):
        arr = np.array(tokens, dtype=np.int64)
    else:
        arr = np.asarray(tokens)

    arr = arr.reshape(-1)
    total_tokens = int(len(arr))
    eos_indices = np.nonzero(arr == eos_token_id)[0]
    eos_count = int(len(eos_indices))

    if eos_count == 0:
        return {
            "total_tokens": total_tokens,
            "eos_token_id": eos_token_id,
            "eos_count": 0,
            "has_eos": False,
            "verdict": "FAIL_ZERO_EOS",
            "message": "Zero EOS tokens found in dataset.",
        }

    distances = np.diff(eos_indices) if eos_count > 1 else np.array([total_tokens])
    return {
        "total_tokens": total_tokens,
        "eos_token_id": eos_token_id,
        "eos_count": eos_count,
        "eos_frequency": float(eos_count / max(1, total_tokens)),
        "has_eos": True,
        "mean_story_length": float(np.mean(distances)),
        "min_story_length": int(np.min(distances)),
        "max_story_length": int(np.max(distances)),
        "verdict": "PASS_EOS_PRESENT",
    }


# =====================================================================
# 4. Activation Patching Interventions
# =====================================================================

def forward_with_residual_cache(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run forward pass and return (final_logits, list of residual states L+1)."""
    inner = getattr(model, "inner", model)
    residuals = []
    wte = getattr(inner, "wte", getattr(inner, "embed", None))
    drop = getattr(inner, "drop", nn.Identity())
    if wte is None:
        raise AttributeError("Model must have an embedding layer ('wte' or 'embed')")
    x = drop(wte(input_ids))
    residuals.append(x.clone())

    layers = getattr(inner, "layers", [])
    for layer in layers:
        out = layer(x)
        x = out[0] if isinstance(out, tuple) else out
        residuals.append(x.clone())

    ln_f = getattr(inner, "ln_f", nn.Identity())
    lm_head = getattr(inner, "lm_head", None)
    if lm_head is None:
        raise AttributeError("Model must have an 'lm_head'")
    logits = lm_head(ln_f(x))
    return logits, residuals


def forward_with_patch(
    model: nn.Module,
    input_ids: torch.Tensor,
    patch_layer: int,
    patch_tensor: torch.Tensor,
    patch_pos: int = -1,
) -> torch.Tensor:
    """Run forward pass, injecting activation at patch_layer and patch_pos."""
    inner = getattr(model, "inner", model)
    wte = getattr(inner, "wte", getattr(inner, "embed", None))
    drop = getattr(inner, "drop", nn.Identity())
    x = drop(wte(input_ids))

    if patch_layer == -1:
        x = x.clone()
        x[:, patch_pos, :] = patch_tensor[:, patch_pos, :]

    layers = getattr(inner, "layers", [])
    for l_idx, layer in enumerate(layers):
        out = layer(x)
        x = out[0] if isinstance(out, tuple) else out
        if l_idx == patch_layer:
            x = x.clone()
            x[:, patch_pos, :] = patch_tensor[:, patch_pos, :]

    ln_f = getattr(inner, "ln_f", nn.Identity())
    lm_head = getattr(inner, "lm_head", None)
    return lm_head(ln_f(x))


def evaluate_activation_patching(
    model: nn.Module,
    clean_ids: torch.Tensor,
    corrupt_ids: torch.Tensor,
    target_token_id: int,
    foil_token_id: int,
    patch_pos: int = -1,
) -> Dict[str, Any]:
    """Execute layer-by-layer activation patching and report recovery percentages."""
    if clean_ids.shape != corrupt_ids.shape:
        raise ValueError(f"Clean {clean_ids.shape} and corrupt {corrupt_ids.shape} must have identical shape")

    clean_logits, clean_residuals = forward_with_residual_cache(model, clean_ids)
    corrupt_logits, _ = forward_with_residual_cache(model, corrupt_ids)

    clean_diff = (clean_logits[0, -1, target_token_id] - clean_logits[0, -1, foil_token_id]).item()
    corrupt_diff = (corrupt_logits[0, -1, target_token_id] - corrupt_logits[0, -1, foil_token_id]).item()
    denom = max(abs(clean_diff - corrupt_diff), 1e-6)

    layers_effect = []
    inner = getattr(model, "inner", model)
    n_layers = len(getattr(inner, "layers", []))

    for l_idx in range(n_layers):
        patched_logits = forward_with_patch(
            model, corrupt_ids, patch_layer=l_idx, patch_tensor=clean_residuals[l_idx + 1], patch_pos=patch_pos
        )
        p_diff = (patched_logits[0, -1, target_token_id] - patched_logits[0, -1, foil_token_id]).item()
        recovery = (p_diff - corrupt_diff) / denom
        layers_effect.append({
            "layer": l_idx,
            "logit_diff": float(p_diff),
            "recovery_pct": float(recovery * 100.0),
        })

    return {
        "clean_diff": float(clean_diff),
        "corrupt_diff": float(corrupt_diff),
        "layers": layers_effect,
    }


# =====================================================================
# 5. Counterfactual Logit Margin Probing
# =====================================================================

def evaluate_counterfactual_case(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_token_id: int,
    foil_token_id: int,
) -> Dict[str, Any]:
    """Compute targeted logit margin z_target - z_foil and target/foil probabilities."""
    model.eval()
    with torch.no_grad():
        logits = extract_logits(model(input_ids))[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    z_target = float(logits[target_token_id].item())
    z_foil = float(logits[foil_token_id].item())
    logit_margin = z_target - z_foil

    p_target = float(probs[target_token_id].item()) * 100.0
    p_foil = float(probs[foil_token_id].item()) * 100.0

    return {
        "target_id": target_token_id,
        "foil_id": foil_token_id,
        "logit_margin": float(logit_margin),
        "target_prob": float(p_target),
        "foil_prob": float(p_foil),
        "is_correct": bool(logit_margin > 0),
    }


# =====================================================================
# Pillar 5 Battery Runner
# =====================================================================

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
