"""Delimited Selective Copying Benchmark Task (Mamba Appendix E.1).

Data tokens are placed at random positions separated by noise tokens.
A delimiter token signals the end of the intake phase and the beginning of the reproduction phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE_TOKEN_ID = 0
DELIMITER_TOKEN_ID = 1
PAD_TOKEN_ID = 2
DATA_TOKEN_START = 3


@dataclass
class SelectiveCopyingConfig:
    seq_len: int = 128
    vocab_size: int = 32
    num_data_tokens: int = 8
    batch_size: int = 32


def generate_batch(
    cfg: SelectiveCopyingConfig,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a delimited selective copying batch.

    Input format:
        [noise, d1, noise, noise, d2, ..., delimiter, pad, pad, pad, ...]
    Supervision:
        Labels are -100 during noise and intake, and require exact reproduction of
        [d1, d2, ..., dK] immediately after the delimiter.
    """
    B, L = cfg.batch_size, cfg.seq_len
    K = cfg.num_data_tokens
    if cfg.vocab_size <= DATA_TOKEN_START + K:
        raise ValueError(
            f"vocab_size={cfg.vocab_size} must be > {DATA_TOKEN_START + K} to accommodate {K} data tokens"
        )
    if L < 2 * K + 4:
        raise ValueError(f"seq_len={L} is too short for {K} data tokens and reproduction window")

    input_ids = torch.full((B, L), NOISE_TOKEN_ID, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)

    # Available data tokens from DATA_TOKEN_START onwards
    data_token_pool = torch.arange(DATA_TOKEN_START, DATA_TOKEN_START + K)

    # Delimiter placed at L - K - 1 so exactly K pad tokens follow it
    delim_pos = L - K - 1

    for b in range(B):
        # Permute data tokens for this sample
        perm = torch.randperm(K, generator=generator)
        sample_data = data_token_pool[perm]

        # Choose K non-overlapping positions strictly before delim_pos
        data_pos = torch.randperm(delim_pos, generator=generator)[:K].sort().values
        input_ids[b, data_pos] = sample_data

        # Place delimiter
        input_ids[b, delim_pos] = DELIMITER_TOKEN_ID

        # Fill output slots with PAD_TOKEN
        input_ids[b, delim_pos + 1:] = PAD_TOKEN_ID

        # Supervise output reproduction: position delim_pos predicts sample_data[0] at delim_pos + 1
        labels[b, delim_pos + 1: delim_pos + 1 + K] = sample_data

    return input_ids.to(device), labels.to(device)


def compute_causal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact match accuracy across selective copying reproduction slots."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not mask.any():
        return 0.0
    preds = shift_logits.argmax(dim=-1)
    return (preds[mask] == shift_labels[mask]).float().mean().item()


def compute_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss across selective copying reproduction slots."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class SelectiveCopyingOracle(nn.Module):
    """Analytical Oracle achieving 100% exact match on delimited selective copying."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        logits = torch.full((B, L, self.vocab_size), -1e4, device=input_ids.device)
        for b in range(B):
            row = input_ids[b].tolist()
            collected = [tok for tok in row if tok >= DATA_TOKEN_START]
            # When delimiter is reached at position p, emit collected[0] at p (predicting p+1),
            # collected[1] at p+1, etc.
            delim_indices = [idx for idx, tok in enumerate(row) if tok == DELIMITER_TOKEN_ID]
            if delim_indices:
                p = delim_indices[0]
                for offset, target_val in enumerate(collected):
                    pos = p + offset
                    if pos < L:
                        logits[b, pos, target_val] = 100.0
        return logits
