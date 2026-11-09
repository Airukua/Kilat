"""BDD Tests for Kilat Benchmark Synthetic Tasks (MQAR, Induction Heads, Selective Copying).

Proves:
1. Analytical Oracles achieve 100% exact match accuracy.
2. Leak-Free Induction Heads: identity shortcuts fail (accuracy = 0%).
3. Delimited Selective Copying: delimiter invariants and causal target layout.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kilat.benchmarking.tasks.induction_heads import (
    QUERY_TOKEN_ID,
    SPECIAL_TOKEN_ID,
    InductionAnalyticalOracle,
    InductionHeadsConfig,
    compute_causal_accuracy as compute_induction_acc,
    generate_batch as generate_induction_batch,
)
from kilat.benchmarking.tasks.mqar import (
    MQARAnalyticalOracle,
    MQARConfig,
    compute_causal_accuracy as compute_mqar_acc,
    generate_batch as generate_mqar_batch,
)
from kilat.benchmarking.tasks.selective_copying import (
    DELIMITER_TOKEN_ID,
    NOISE_TOKEN_ID,
    PAD_TOKEN_ID,
    SelectiveCopyingConfig,
    SelectiveCopyingOracle,
    compute_causal_accuracy as compute_copying_acc,
    generate_batch as generate_copying_batch,
)
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_mqar_batch_invariants_and_oracle():
    with Scenario("Benchmark: MQAR Batch Generation & Analytical Oracle"):
        with Given("an MQAR configuration with B=4, N=64, d=4, V=128"):
            cfg = MQARConfig(seq_len=64, vocab_size=128, num_kv_pairs=4, batch_size=4)
            gen = torch.Generator().manual_seed(42)

        with When("generating a batch and evaluating with the MQAR Analytical Oracle"):
            x, y = generate_mqar_batch(cfg, generator=gen)
            oracle = MQARAnalyticalOracle(vocab_size=cfg.vocab_size)
            logits = oracle(x)
            acc = compute_mqar_acc(logits, y)

        with Then("the generated tensors match expected shapes [B, N]"):
            assert x.shape == (4, 64)
            assert y.shape == (4, 64)
        with And("exactly d answer slots are unmasked per sequence"):
            unmasked_per_row = (y != -100).sum(dim=1)
            assert (unmasked_per_row == 4).all().item()
        with And("the Analytical Oracle achieves exactly 100% accuracy"):
            assert acc == 1.0


def test_induction_heads_leak_free_and_oracle():
    with Scenario("Benchmark: Induction Heads Leak-Free Invariant & Oracle Proof"):
        with Given("an Induction Heads configuration with B=4, N=64, V=64"):
            cfg = InductionHeadsConfig(vocab_size=64, seq_len=64, batch_size=4)
            gen = torch.Generator().manual_seed(101)

        with When("generating a batch and checking token identity at target positions"):
            x, y = generate_induction_batch(cfg, generator=gen)
            oracle = InductionAnalyticalOracle(vocab_size=cfg.vocab_size)
            oracle_logits = oracle(x)
            oracle_acc = compute_induction_acc(oracle_logits, y)

        with Then("target position in input_ids contains QUERY_TOKEN_ID, never target token A"):
            target_mask = y != -100
            target_inputs = x[target_mask]
            target_labels = y[target_mask]
            assert (target_inputs == QUERY_TOKEN_ID).all().item()
            # None of the input tokens at the target position equal the ground truth label
            assert not (target_inputs == target_labels).any().item()
        with And("the Analytical Oracle achieves exactly 100% exact match"):
            assert oracle_acc == 1.0


def test_induction_heads_negative_identity_shortcut_fails():
    with Scenario("Benchmark: Induction Heads Negative Test - Identity Shortcut Collapse"):
        with Given("a corrupt model that simply emits the token present at the query position"):
            class IdentityShortcutModel(nn.Module):
                def __init__(self, vocab_size):
                    super().__init__()
                    self.vocab_size = vocab_size

                def forward(self, input_ids):
                    B, L = input_ids.shape
                    logits = torch.full((B, L, self.vocab_size), -1e4)
                    for b in range(B):
                        for t in range(L):
                            tok = input_ids[b, t].item()
                            logits[b, t, tok] = 100.0
                    return logits

            cfg = InductionHeadsConfig(vocab_size=64, seq_len=64, batch_size=4)
            x, y = generate_induction_batch(cfg, generator=torch.Generator().manual_seed(777))
            corrupt_model = IdentityShortcutModel(vocab_size=cfg.vocab_size)

        with When("evaluating the identity shortcut model"):
            logits = corrupt_model(x)
            acc = compute_induction_acc(logits, y)

        with Then("the accuracy collapses to exactly 0%, proving the absence of input leakage"):
            assert acc == 0.0


def test_selective_copying_delimiters_and_oracle():
    with Scenario("Benchmark: Delimited Selective Copying Invariants & Oracle"):
        with Given("a Selective Copying configuration with B=4, N=64, K=6, V=32"):
            cfg = SelectiveCopyingConfig(seq_len=64, vocab_size=32, num_data_tokens=6, batch_size=4)
            gen = torch.Generator().manual_seed(999)

        with When("generating a batch and running the Selective Copying Oracle"):
            x, y = generate_copying_batch(cfg, generator=gen)
            oracle = SelectiveCopyingOracle(vocab_size=cfg.vocab_size)
            logits = oracle(x)
            acc = compute_copying_acc(logits, y)

        with Then("exactly one DELIMITER_TOKEN is placed per sequence"):
            delims = (x == DELIMITER_TOKEN_ID).sum(dim=1)
            assert (delims == 1).all().item()
        with And("exactly K answer slots are unmasked following the delimiter"):
            unmasked = (y != -100).sum(dim=1)
            assert (unmasked == 6).all().item()
        with And("the Selective Copying Oracle achieves exactly 100% accuracy"):
            assert acc == 1.0


def test_selective_copying_negative_delimiter_sabotage():
    with Scenario("Benchmark: Selective Copying Negative Test - Delimiter Sabotage"):
        with Given("a valid batch where delimiter tokens are corrupted into noise tokens"):
            cfg = SelectiveCopyingConfig(seq_len=64, vocab_size=32, num_data_tokens=6, batch_size=4)
            x, y = generate_copying_batch(cfg, generator=torch.Generator().manual_seed(333))
            # Sabotage: erase delimiter token
            x[x == DELIMITER_TOKEN_ID] = NOISE_TOKEN_ID
            oracle = SelectiveCopyingOracle(vocab_size=cfg.vocab_size)

        with When("evaluating the Oracle on the sabotaged sequence"):
            logits = oracle(x)
            acc = compute_copying_acc(logits, y)

        with Then("the Oracle fails to trigger reproduction and accuracy drops to 0%"):
            assert acc == 0.0
