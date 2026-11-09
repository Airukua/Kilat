"""
BDD-style tests for tools/synthetic/mqar.py (Multi-Query Associative Recall Benchmark).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.synthetic.mqar import FILLER_TOKEN, evaluate, make_mqar_batch
from kilat import KilatConfig, KilatTransformer
from tests.shared.bdd import Scenario, Given, When, Then, And


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OracleMQAR(nn.Module):
    """Analytical Oracle: performs exact key-value lookup from context prefix."""

    def __init__(self, vocab_size: int, num_kv: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_kv = num_kv

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, N = input_ids.shape
        logits = torch.full((B, N, self.vocab_size), -100.0, device=input_ids.device)
        ctx = 2 * self.num_kv
        for b in range(B):
            kv_map = {}
            for i in range(0, ctx, 2):
                k = input_ids[b, i].item()
                v = input_ids[b, i + 1].item()
                kv_map[k] = v
            for t in range(ctx, N):
                tok = input_ids[b, t].item()
                if tok in kv_map:
                    logits[b, t, kv_map[tok]] = 100.0
                else:
                    logits[b, t, 0] = 0.0

        class Output:
            pass

        out = Output()
        out.logits = logits
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out.loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            out.loss = None
        return out


class RandomPredictor(nn.Module):
    """Baseline uniform predictor: emits constant zero logits."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, N = input_ids.shape
        logits = torch.zeros((B, N, self.vocab_size), device=input_ids.device)

        class Output:
            pass

        out = Output()
        out.logits = logits
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out.loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return out


def test_mqar_batch_generation_invariants():
    with Scenario("Tools: MQAR Batch Generation & Tensor Invariants"):
        with Given("batch configuration B=4, N=64, d=4 kv-pairs, V=256"):
            B, N, d, V = 4, 64, 4, 256
            g = torch.Generator().manual_seed(42)

            with When("make_mqar_batch generates input and label tensors"):
                x, y = make_mqar_batch(B, N, d, V, g)

                with Then("the tensors have shape [B, N] and long dtype"):
                    assert x.shape == (B, N)
                    assert y.shape == (B, N)
                    assert x.dtype == torch.long
                    assert y.dtype == torch.long

                with And("prefix keys and values lie in strictly disjoint token intervals"):
                    half = V // 2
                    ctx = 2 * d
                    for b in range(B):
                        keys = x[b, 0:ctx:2].tolist()
                        vals = x[b, 1:ctx:2].tolist()
                        assert len(set(keys)) == d
                        for k in keys:
                            assert 1 <= k < half
                        for v in vals:
                            assert half <= v < V

                with And("exactly d answer slots are unmasked in labels"):
                    for b in range(B):
                        active_labels = (y[b] != -100).sum().item()
                        assert active_labels == d


def test_mqar_oracle_accuracy_and_loss(device: torch.device):
    with Scenario("Tools: Analytical Oracle Associative Recall Proof"):
        with Given("an analytical Oracle model and held-out evaluation set"):
            vocab_size = 256
            seq_len = 64
            num_kv = 4
            batch_size = 16
            g = torch.Generator().manual_seed(42)
            eval_set = [make_mqar_batch(batch_size, seq_len, num_kv, vocab_size, g) for _ in range(4)]
            oracle = OracleMQAR(vocab_size, num_kv).to(device)

            with When("evaluating Oracle recall accuracy and cross-entropy loss"):
                acc = evaluate(oracle, eval_set, device, amp_dtype=None)
                x_test, y_test = eval_set[0]
                loss = oracle(x_test.to(device), labels=y_test.to(device)).loss.item()

                with Then("the Oracle achieves exactly 100% exact-match accuracy"):
                    assert acc == 1.0

                with And("the cross-entropy loss is near-zero (< 1e-4)"):
                    assert loss < 1e-4


def test_mqar_single_batch_overfit(device: torch.device):
    with Scenario("Tools: KilatTransformer Single Batch Overfit"):
        with Given("a 2-layer KilatTransformer and single MQAR training batch"):
            vocab_size = 256
            seq_len = 64
            num_kv = 4
            batch_size = 16
            g = torch.Generator().manual_seed(100)
            batch_x, batch_y = make_mqar_batch(batch_size, seq_len, num_kv, vocab_size, g)
            eval_set = [(batch_x, batch_y)]

            cfg = KilatConfig(
                vocab_size=vocab_size,
                n_embd=64,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                latent_dim=32,
                max_position_embeddings=seq_len,
                use_rope=True,
                rope_head_dim=8,
            )
            model = KilatTransformer(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
            x = batch_x.to(device)
            y = batch_y.to(device)

            with When("optimizing the batch over 150 gradient steps"):
                for _ in range(150):
                    model.train()
                    loss = model(x, labels=y).loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                final_acc = evaluate(model, eval_set, device, amp_dtype=None)

                with Then("the final training accuracy reaches >= 95%"):
                    assert final_acc >= 0.95


def test_mqar_negative_invalid_sequence_length():
    with Scenario("Tools: MQAR Negative Testing - Invalid Sequence Length"):
        with Given("parameters where seq_len is too short for KV context and queries"):
            g = torch.Generator().manual_seed(42)

            with When("generating a batch with insufficient sequence length"):
                with Then("a ValueError is raised"):
                    with pytest.raises(ValueError, match="too short"):
                        make_mqar_batch(batch_size=2, seq_len=16, num_kv_pairs=8, vocab_size=256, generator=g)


def test_mqar_negative_random_guessing(device: torch.device):
    with Scenario("Tools: MQAR Negative Testing - Random Predictor Entropy & Accuracy"):
        with Given("a baseline predictor emitting uniform zero logits"):
            vocab_size = 256
            seq_len = 64
            num_kv = 4
            batch_size = 16
            g = torch.Generator().manual_seed(200)
            eval_set = [make_mqar_batch(batch_size, seq_len, num_kv, vocab_size, g) for _ in range(4)]
            model = RandomPredictor(vocab_size).to(device)

            with When("evaluating cross-entropy loss and accuracy on held-out set"):
                acc = evaluate(model, eval_set, device, amp_dtype=None)
                x_test, y_test = eval_set[0]
                loss = model(x_test.to(device), labels=y_test.to(device)).loss.item()
                expected_loss = math.log(vocab_size)

                with Then("the loss matches theoretical entropy ln(vocab_size)"):
                    assert abs(loss - expected_loss) < 1e-3

                with And("the evaluation accuracy is near zero (< 5%)"):
                    assert acc < 0.05


def test_mqar_negative_off_by_one_sabotage(device: torch.device):
    with Scenario("Tools: MQAR Negative Testing - Off-By-One Target Sabotage"):
        with Given("an analytical Oracle and target labels shifted by 1 slot"):
            vocab_size = 256
            seq_len = 64
            num_kv = 4
            batch_size = 16
            g = torch.Generator().manual_seed(300)
            eval_set = [make_mqar_batch(batch_size, seq_len, num_kv, vocab_size, g) for _ in range(4)]
            oracle = OracleMQAR(vocab_size, num_kv).to(device)
            sabotaged_set = [(x, torch.roll(y, shifts=1, dims=1)) for x, y in eval_set]

            with When("computing recall accuracy on the sabotaged labels"):
                sab_acc = evaluate(oracle, sabotaged_set, device, amp_dtype=None)

                with Then("the evaluation accuracy collapses to exactly 0%"):
                    assert sab_acc == 0.0


def test_mqar_negative_causal_leakage(device: torch.device):
    with Scenario("Tools: MQAR Negative Testing - Causal Masking & Future Information Leakage"):
        with Given("a KilatTransformer model and input sequence with split pos 12"):
            vocab_size = 256
            seq_len = 64
            num_kv = 4
            batch_size = 8
            g = torch.Generator().manual_seed(400)
            x, _ = make_mqar_batch(batch_size, seq_len, num_kv, vocab_size, g)
            x = x.to(device)

            cfg = KilatConfig(
                vocab_size=vocab_size,
                n_embd=64,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                latent_dim=32,
                max_position_embeddings=seq_len,
                use_rope=True,
                rope_head_dim=8,
            )
            model = KilatTransformer(cfg).to(device).eval()

            split_pos = 12
            x_mutated = x.clone()
            x_mutated[:, split_pos + 1 :] = torch.randint(
                1, vocab_size, x_mutated[:, split_pos + 1 :].shape, device=device
            )

            with When("forward passes are computed for original and future-mutated inputs"):
                with torch.no_grad():
                    out_orig = model(x).logits
                    out_mut = model(x_mutated).logits

                diff_past = (out_orig[:, : split_pos + 1, :] - out_mut[:, : split_pos + 1, :]).abs().max().item()
                diff_future = (out_orig[:, split_pos + 1 :, :] - out_mut[:, split_pos + 1 :, :]).abs().max().item()

                with Then("past and present logits remain strictly unchanged (diff < 1e-5)"):
                    assert diff_past < 1e-5

                with And("future logits reflect the mutated tokens (diff > 1e-3)"):
                    assert diff_future > 1e-3
