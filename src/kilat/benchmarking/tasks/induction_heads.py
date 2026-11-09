"""Leak-Free Induction Heads Task (Olsson et al., 2022; Mamba Appendix E.1).

A special token S appears twice:
    Sequence: [..., S, A, ..., S, <QUERY>]
    Target: The model at position 'second_pos' must predict token A.
    Crucial Causal Invariant: The position 'second_pos + 1' contains a filler/query token,
    NEVER token A itself. This prevents the identity auto-encoding shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECIAL_TOKEN_ID = 1
QUERY_TOKEN_ID = 2


@dataclass
class InductionHeadsConfig:
    vocab_size: int = 64
    seq_len: int = 128
    batch_size: int = 32


def generate_batch(
    cfg: InductionHeadsConfig,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch for the induction heads task.

    Returns:
        input_ids: LongTensor [B, L]
        labels: LongTensor [B, L], where only the slot after the second special token
                contains the ground-truth target token A, and all other slots are -100.
    """
    B, L = cfg.batch_size, cfg.seq_len
    if L < 8:
        raise ValueError(f"seq_len={L} is too short for induction heads benchmark (min 8)")
    if cfg.vocab_size < 8:
        raise ValueError("vocab_size must be >= 8 to allow distinct special/query tokens")

    # Tokens 3..vocab_size-1 are random data tokens
    input_ids = torch.randint(
        low=3,
        high=cfg.vocab_size,
        size=(B, L),
        generator=generator,
        dtype=torch.long,
    )
    labels = torch.full((B, L), -100, dtype=torch.long)

    for b in range(B):
        # Choose two non-overlapping positions for special token
        first_pos = int(torch.randint(0, L // 2 - 2, (1,), generator=generator).item())
        second_pos = int(torch.randint(L // 2, L - 2, (1,), generator=generator).item())

        input_ids[b, first_pos] = SPECIAL_TOKEN_ID
        key_token = input_ids[b, first_pos + 1].clone()

        input_ids[b, second_pos] = SPECIAL_TOKEN_ID
        # The query position receives a generic query token, NOT key_token
        input_ids[b, second_pos + 1] = QUERY_TOKEN_ID

        # Supervision: The model at 'second_pos' predicts 'key_token' for 'second_pos + 1'
        labels[b, second_pos + 1] = key_token

    return input_ids.to(device), labels.to(device)


def compute_causal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact match accuracy at the induction target position."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return 0.0
    preds = shift_logits.argmax(dim=-1)
    return (preds[mask] == shift_labels[mask]).float().mean().item()


def compute_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss at the induction target position."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def extrapolation_lengths(train_len: int = 128, max_len: int = 4096) -> List[int]:
    """Generate exponential extrapolation evaluation sequence lengths."""
    lengths = []
    L = 64
    while L <= max_len:
        lengths.append(L)
        L *= 2
    if train_len not in lengths:
        lengths.append(train_len)
        lengths.sort()
    return lengths


class InductionAnalyticalOracle(nn.Module):
    """Analytical Oracle achieving 100% exact match on induction heads without training."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        logits = torch.full((B, L, self.vocab_size), -1e4, device=input_ids.device)
        for b in range(B):
            row = input_ids[b].tolist()
            first_key = None
            for t in range(L - 1):
                if row[t] == SPECIAL_TOKEN_ID:
                    if first_key is None:
                        first_key = row[t + 1]
                    else:
                        # At second special token, emit high logit for first_key
                        logits[b, t, first_key] = 100.0
        return logits
