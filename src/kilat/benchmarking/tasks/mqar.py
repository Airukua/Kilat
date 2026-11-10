"""Zoology Multi-Query Associative Recall (MQAR) Benchmark Task.

Reference:
    Arora, Eyuboglu et al., "Zoology: Measuring and Improving Recall in Efficient Language Models", ICLR 2024.
    Implementation aligned 1:1 with Stanford Hazy Research Zoology repository (zoology/data/multiquery_ar.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
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
    power_a: float = 0.01
    random_non_queries: bool = False
    num_passes: int = 1


def generate_dataset(
    cfg: MQARConfig,
    num_examples: int,
    seed: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a full MQAR dataset aligned 1:1 with Stanford Zoology multiquery_ar.

    Structure:
        Context: [k1, v1, k2, v2, ..., kd, vd] (length = 2 * d * num_passes)
        Query region: queries placed according to power-law gap distribution.
        Inputs: examples[:, :-1] of length seq_len.
        Labels: labels[:, 1:] of length seq_len, where position t contains the target
                value for the query key present at input position t, and -100 elsewhere.
    """
    assert cfg.seq_len % 2 == 0, "seq_len must be even"
    context_size = cfg.num_kv_pairs * 2 * cfg.num_passes
    assert (
        context_size + cfg.num_kv_pairs * 2 <= cfg.seq_len
    ), f"seq_len {cfg.seq_len} too short for {cfg.num_kv_pairs} KV pairs (requires >= {context_size + cfg.num_kv_pairs * 2})"
    assert (
        cfg.vocab_size // 2 > cfg.num_kv_pairs
    ), f"vocab_size {cfg.vocab_size} too small to draw {cfg.num_kv_pairs} unique keys"

    rng = np.random.RandomState(seed)

    key_vocab_size = cfg.vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, cfg.vocab_size)

    keys = np.zeros((num_examples, cfg.num_kv_pairs), dtype=np.int64)
    values = np.zeros((num_examples, cfg.num_kv_pairs), dtype=np.int64)
    for i in range(num_examples):
        keys[i] = rng.choice(key_choices, size=cfg.num_kv_pairs, replace=False)
        values[i] = rng.choice(value_choices, size=cfg.num_kv_pairs, replace=False)

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    if cfg.num_passes > 1:
        kvs = np.tile(kvs, (1, cfg.num_passes))

    space = (cfg.seq_len - context_size) // 2
    p = cfg.power_a * np.arange(1, space + 1) ** (cfg.power_a - 1)
    p = p / p.sum()

    gaps = np.zeros((num_examples, cfg.num_kv_pairs), dtype=np.int64)
    slots_arr = np.arange(space, dtype=np.int64)
    for i in range(num_examples):
        gaps[i] = rng.choice(slots_arr, size=cfg.num_kv_pairs, replace=False, p=p)

    queries = np.zeros((num_examples, cfg.seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, cfg.seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs_t = torch.tensor(examples[:, :-1], dtype=torch.long)
    labels_t = torch.tensor(labels[:, 1:], dtype=torch.long)

    if cfg.random_non_queries:
        zero_mask = inputs_t == 0
        rand_tokens = torch.from_numpy(rng.randint(0, cfg.vocab_size, size=inputs_t.shape))
        inputs_t[zero_mask] = rand_tokens[zero_mask]

    return inputs_t.to(device), labels_t.to(device)


def generate_batch(
    cfg: MQARConfig,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a single MQAR batch aligned with Stanford Zoology."""
    if generator is not None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    else:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    return generate_dataset(cfg, num_examples=cfg.batch_size, seed=seed, device=device)


def compute_causal_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact match accuracy at target query positions (Zoology convention)."""
    mask = labels != -100
    if not mask.any():
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()
    return correct / max(1, total)


def compute_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss at target query positions (Zoology convention)."""
    return F.cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        labels.contiguous().view(-1),
        ignore_index=-100,
    )


class MQARAnalyticalOracle(nn.Module):
    """Analytical non-parametric Oracle for MQAR verification (achieves 100% exact match)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        logits = torch.full((B, L, self.vocab_size), -1e4, device=input_ids.device)
        for b in range(B):
            row = input_ids[b].tolist()
            kv_map = {}
            for i in range(0, L - 1, 2):
                k, v = row[i], row[i + 1]
                if k > 0 and v > 0 and k < self.vocab_size // 2 and v >= self.vocab_size // 2:
                    kv_map[k] = v
                else:
                    break
            for t in range(L):
                tok = row[t]
                if tok in kv_map:
                    target_v = kv_map[tok]
                    logits[b, t, target_v] = 100.0
        return logits
