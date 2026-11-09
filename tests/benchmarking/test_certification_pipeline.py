"""BDD Tests for Kilat Model Architecture Certification Pipeline (KMCP).

Verifies:
1. Pillar 1 Mechanical Correctness on Kilat architecture.
2. Pillar 4 Efficiency and speedup comparison against Reference Attention.
3. Full Certification Dossier generation (Markdown + JSON) with 5-pillar scorecard.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kilat.benchmarking.harness import CertificationHarness
from kilat.benchmarking.pillars.p1_correctness import CorrectnessConfig, run_pillar1
from kilat.benchmarking.pillars.p4_efficiency import EfficiencyConfig, run_pillar4
from kilat.benchmarking.pillars.p5_diagnostics import DiagnosticsConfig, run_pillar5
from kilat.benchmarking.registry import build_model
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_pillar1_mechanical_correctness_kilat():
    with Scenario("Benchmark: Pillar 1 Mechanical Correctness on Kilat-v2"):
        with Given("a micro Kilat-v2 architecture builder and correctness config"):
            cfg = CorrectnessConfig(
                seq_len=16,
                vocab_size=32,
                batch_size=2,
                overfit_steps=120,
                overfit_loss_threshold=0.10,
                lr=5e-3,
                seeds=[0],
                device="cpu",
            )
            def builder(**kw):
                params = {"vocab_size": 32, "d_model": 32, "n_layers": 2, "n_heads": 2}
                params.update(kw)
                return build_model("kilat-v2", **params)

        with When("executing Pillar 1 battery"):
            report = run_pillar1(builder, cfg)

        with Then("causal future blindness test passes with diff <= 1e-5"):
            assert report["causal_future_blindness"]["passed"] is True
            assert report["causal_future_blindness"]["max_past_leak_diff"] <= 1e-5
        with And("gradient connectivity test passes with 0 NaN, 0 Inf, and 0 dead gradients"):
            assert report["gradient_connectivity"]["passed"] is True
            assert report["gradient_connectivity"]["n_nan"] == 0
            assert report["gradient_connectivity"]["n_dead"] == 0
        with And("the overall Pillar 1 status is passed"):
            assert report["passed"] is True


def test_pillar4_efficiency_and_speedup():
    with Scenario("Benchmark: Pillar 4 Efficiency Dynamics & Comparative Speedup"):
        with Given("model builders for Kilat-v2 and Reference Attention"):
            cfg = EfficiencyConfig(
                seq_lens=[16, 32],
                batch_size=2,
                n_warmup=1,
                n_iters=3,
                n_decode_steps=4,
                vocab_size=32,
                device="cpu",
            )
            def model_builder(**kw):
                params = {"vocab_size": 32, "d_model": 32, "n_layers": 2, "n_heads": 2}
                params.update(kw)
                return build_model("kilat-v2", **params)

            def base_builder(**kw):
                params = {"vocab_size": 32, "d_model": 32, "n_layers": 2, "n_heads": 2}
                params.update(kw)
                return build_model("reference-attention", **params)

        with When("running Pillar 4 comparative efficiency benchmark"):
            report = run_pillar4(model_builder, cfg, baseline_builder=base_builder)

        with Then("the sweep contains prefill tokens/sec for both sequence lengths"):
            assert len(report["seq_len_sweep"]) == 2
            assert report["seq_len_sweep"][0]["prefill"]["prefill_tokens_per_sec"] > 0
        with And("the decode benchmark records tokens/sec and ms/token"):
            assert report["decode"]["decode_tokens_per_sec"] > 0
            assert report["decode"]["decode_ms_per_token"] > 0
        with And("comparative speedup metrics are calculated against baseline"):
            assert "baseline_comparison" in report
            assert len(report["baseline_comparison"]["sweep"]) == 2
            assert report["baseline_comparison"]["sweep"][0]["speedup"] > 0


def test_full_certification_harness_execution(tmp_path: Path):
    with Scenario("Benchmark: End-to-End Certification Harness Dossier Generation"):
        with Given("a smoke test certification configuration and temporary output directory"):
            raw_cfg = {
                "pillar1": {
                    "seq_len": 16,
                    "vocab_size": 64,
                    "batch_size": 2,
                    "overfit_steps": 30,
                    "overfit_loss_threshold": 0.20,
                    "seeds": [0],
                    "device": "cpu",
                },
                "pillar2": {
                    "task": "mqar",
                    "context_lengths": [32],
                    "seeds": [0],
                    "train_steps": 20,
                    "vocab_size": 64,
                    "batch_size": 8,
                    "device": "cpu",
                },
                "pillar3": {
                    "seq_lens": [32],
                    "token_budgets": [500],
                    "eval_every_tokens": 250,
                    "vocab_size": 64,
                    "batch_size": 4,
                    "device": "cpu",
                },
                "pillar4": {
                    "seq_lens": [16],
                    "batch_size": 2,
                    "n_warmup": 1,
                    "n_iters": 2,
                    "n_decode_steps": 2,
                    "vocab_size": 64,
                    "device": "cpu",
                },
                "pillar5": {
                    "seq_len": 32,
                    "vocab_size": 64,
                    "batch_size": 2,
                    "device": "cpu",
                    "buckets": [0, 16, 32],
                },
            }
            harness = CertificationHarness(raw_cfg)

        with When("certifying model 'reference-attention' across all 5 pillars"):
            results = harness.certify_model(
                model_name="reference-attention",
                output_dir=str(tmp_path),
                pillars=["p1", "p2", "p3", "p4", "p5"],
            )

        with Then("all 5 pillars are executed and recorded in results"):
            assert "p1" in results
            assert "p2" in results
            assert "p3" in results
            assert "p4" in results
            assert "p5" in results
        with And("the certification dossier files are written to disk"):
            md_file = tmp_path / "reference-attention_certification_dossier.md"
            json_file = tmp_path / "reference-attention_certification_raw.json"
            assert md_file.is_file()
            assert json_file.is_file()
            content = md_file.read_text(encoding="utf-8")
            assert "Certification Scorecard" in content
            assert "Pillar 1" in content
            assert "Pillar 2" in content
            assert "Pillar 3" in content
            assert "Pillar 4" in content
            assert "Pillar 5" in content
