"""Zoology Multi-Query Associative Recall (MQAR) Benchmark Task.

Reference: Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models", ICLR 2024.
Enforces strict causal alignment: predictions at position p predict labels at position p + 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FILLER_TOKEN = 0


@dataclass
class MQARConfig:
    seq_len: int = 128
    vocab_size: int = 256
    num_kv_pairs: int = 8
    batch_size: int = 32


def generate_batch(
    cfg: MQARConfig,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate an MQAR batch with non-overlapping keys and values.

    Input sequence structure:
        [k1, v1, k2, v2, ..., kd, vd, 0, 0, q1, 0, 0, q2, 0, ...]
    Labels:
        -100 everywhere except at query_pos + 1, where the true value resides.
    """
    B, L = cfg.batch_size, cfg.seq_len
    d = cfg.num_kv_pairs
    ctx_len = 2 * d
    query_region = L - ctx_len
    if query_region < 2 * d:
        raise ValueError(
            f"seq_len={L} is too short for {d} KV pairs and {d} queries (requires >= {4 * d})"
        )

    half = cfg.vocab_size // 2
    if half - 1 < d:
        raise ValueError(f"vocab_size={cfg.vocab_size} too small to draw {d} unique keys")

    input_ids = torch.full((B, L), FILLER_TOKEN, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)

    # Unique keys from [1, half) and values from [half, vocab_size)
    key_pool = torch.rand(B, half - 1, generator=generator).argsort(dim=1)[:, :d] + 1
    values = torch.randint(half, cfg.vocab_size, (B, d), generator=generator)

    input_ids[:, 0:ctx_len:2] = key_pool
    input_ids[:, 1:ctx_len:2] = values

    # Sample query positions (even offsets in query region)
    n_slots = query_region // 2
    slot_scores = torch.rand(B, n_slots, generator=generator)
    slots = slot_scores.argsort(dim=1)[:, :d]
    query_pos = ctx_len + 2 * slots
    order = torch.rand(B, d, generator=generator).argsort(dim=1)

    rows = torch.arange(B).unsqueeze(1).expand(-1, d)
    q_keys = torch.gather(key_pool, 1, order)
    q_vals = torch.gather(values, 1, order)

    input_ids[rows, query_pos] = q_keys
    labels[rows, query_pos + 1] = q_vals

    return input_ids.to(device), labels.to(device)


def compute_causal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute exact match accuracy using standard causal shift (logits[:-1] vs labels[1:])."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return 0.0
    preds = shift_logits.argmax(dim=-1)
    correct = (preds[mask] == shift_labels[mask]).float().sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def compute_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss using standard causal shift."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class MQARAnalyticalOracle(nn.Module):
    """Analytical non-parametric Oracle for MQAR verification (proves task solvable with 100% accuracy)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        logits = torch.full((B, L, self.vocab_size), -1e4, device=input_ids.device)
        for b in range(B):
            row = input_ids[b].tolist()
            kv_map = {}
            # Reconstruct KV prefix (stops when non-key encountered or query region)
            for i in range(0, L - 1, 2):
                k, v = row[i], row[i + 1]
                if k != FILLER_TOKEN and v != FILLER_TOKEN:
                    kv_map[k] = v
                else:
                    break
            # Predict values immediately following query keys
            for t in range(L - 1):
                tok = row[t]
                if tok in kv_map:
                    target_v = kv_map[tok]
                    logits[b, t, target_v] = 100.0
        return logits
