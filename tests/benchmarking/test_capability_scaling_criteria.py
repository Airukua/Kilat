"""
BDD tests for non-vacuous Pillar 2 / Pillar 3 certification criteria.

Previously Pillar 3 used i.i.d. Zipf tokens (context useless) with pass criterion ppl < 1000, which an untrained
model satisfies at V=256; Pillar 2 judged MQAR only at the shortest context with a below-chance smoke threshold.
"""

import math

import torch
import torch.nn as nn

from kilat.benchmarking.pillars import p2_capability
from kilat.benchmarking.pillars.p2_capability import CapabilityConfig, run_pillar2
from kilat.benchmarking.pillars.p3_scaling import (
    ScalingConfig,
    _default_synthetic_stream,
    markov_entropies,
    run_pillar3,
)
from kilat.benchmarking.registry import build_model
from tests.shared.bdd import And, Given, Scenario, Then, When


class _NoMixing(nn.Module):
    """Per-token lookup model: can learn P(next | current) but has no access to context."""

    def __init__(self, vocab_size: int, **_):
        super().__init__()
        self.e = nn.Embedding(vocab_size, 32)
        self.h = nn.Linear(32, vocab_size)

    def forward(self, x):
        return self.h(self.e(x))


def test_markov_stream_requires_context():
    with Scenario("Benchmark: Pillar 3 Synthetic Data Requires Context"):
        with Given("the default interleaved Markov stream"):
            h_uni, h_cond = markov_entropies(64, branching=4, chain_seed=1234)
            with Then("the Bayes floor is far below the context-free floor, which is below ln V"):
                assert h_cond < 0.5 * h_uni
                assert h_uni <= math.log(64) + 1e-9
            x, y = next(_default_synthetic_stream(64, 33, 4, "cpu"))
            with And("targets are causally aligned with no wrap-around"):
                assert torch.equal(x[:, 1:], y[:, :-1])
                assert x.shape == y.shape == (4, 33)


def test_pillar3_is_not_vacuous_and_discriminates_context_use():
    with Scenario("Benchmark: Pillar 3 Fails Untrained and No-Mixing Models"):
        with Given("an untrained Kilat model (token budget of a single batch)"):
            cfg = ScalingConfig(seq_lens=[32], token_budgets=[1], eval_every_tokens=10**9, vocab_size=64, batch_size=4)
            report = run_pillar3(lambda **kw: build_model("kilat-macro-synergy", **kw), cfg)
            with Then("it FAILS instead of passing a perplexity bound"):
                assert report["passed"] is False
                assert report["runs"][0]["gap_closed"] < 0.1

        with Given("a model with no sequence mixing trained on the same data"):
            cfg = ScalingConfig(seq_lens=[32], token_budgets=[40_000], eval_every_tokens=20_000,
                                vocab_size=64, batch_size=16, lr=3e-3)
            report = run_pillar3(lambda **kw: _NoMixing(**kw), cfg)
            with When("it converges to the context-free floor"):
                with Then("it cannot close the gap and FAILS"):
                    assert report["passed"] is False
                    assert report["runs"][0]["gap_closed"] < 0.2
                with And("the report records both entropy floors"):
                    assert report["data"]["unigram_entropy_nats"] > report["data"]["conditional_entropy_nats"]


def test_pillar2_requires_every_context_length(monkeypatch):
    with Scenario("Benchmark: Pillar 2 Pass Requires Recall at Every Context Length"):
        fake = {64: 0.99, 128: 0.40}

        def _fake_best(model_builder, cfg, seq_len, seed):
            return {"accuracy": fake[seq_len], "loss": 0.1, "best_lr": cfg.lr}

        monkeypatch.setattr(p2_capability, "_best_over_lrs", _fake_best)
        with Given("an architecture that recalls at 64 tokens but collapses at 128"):
            cfg = CapabilityConfig(context_lengths=[64, 128], seeds=[0], pass_threshold=0.9)
            report = run_pillar2(lambda **kw: None, cfg)
            with Then("Pillar 2 FAILS and reports the failing length"):
                assert report["passed"] is False
                assert report["per_length_passed"] == {"64": True, "128": False}

        with Given("an architecture that recalls at every length"):
            fake[128] = 0.95
            report = run_pillar2(lambda **kw: None, cfg)
            with Then("Pillar 2 passes"):
                assert report["passed"] is True


def test_pillar2_best_of_lr_sweep(monkeypatch):
    with Scenario("Benchmark: Pillar 2 Zoology Best-of-LR Protocol"):
        def _fake_single(model_builder, cfg, seq_len, seed, lr=None):
            return {"accuracy": {1e-3: 0.2, 3e-3: 0.97, 1e-2: 0.5}[lr], "loss": 0.1}

        monkeypatch.setattr(p2_capability, "_train_and_eval_single", _fake_single)
        with Given("three candidate learning rates"):
            cfg = CapabilityConfig(context_lengths=[64], seeds=[0], lrs=[1e-3, 3e-3, 1e-2], pass_threshold=0.9)
            report = run_pillar2(lambda **kw: None, cfg)
            with Then("the best LR per cell is selected and recorded"):
                assert report["passed"] is True
                assert report["grid"]["64"]["best_lr"]["mean"] == 3e-3
                assert report["lrs"] == [1e-3, 3e-3, 1e-2]
