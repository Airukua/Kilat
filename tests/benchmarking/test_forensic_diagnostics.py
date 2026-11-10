"""BDD Tests for Kilat Forensic Diagnostics & Mechanistic Probing in Pillar 5.

Verifies:
1. EOS Boundary Auditing (delimiters presence and spacing).
2. N-Gram Contamination Detection (leakage between splits).
3. Counterfactual Logit Margin Probing (margin z_target - z_foil).
4. Residual Stream Extraction & Causal Activation Patching.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from kilat.benchmarking.pillars.p5_diagnostics import (
    audit_eos_boundaries,
    evaluate_activation_patching,
    evaluate_counterfactual_case,
    evaluate_ngram_contamination,
    forward_with_residual_cache,
)
from kilat.benchmarking.registry import build_model
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_eos_boundary_audit_and_zero_detection():
    with Scenario("Benchmark: EOS Document Delimiter Audit & Zero Boundary Detection"):
        with Given("a token sequence containing periodic EOS delimiter tokens (id=50256)"):
            tokens_with_eos = [10, 20, 30, 50256, 40, 50, 50256, 60, 50256]
            tokens_without_eos = [10, 20, 30, 40, 50, 60]

        with When("auditing delimiter distributions"):
            report_clean = audit_eos_boundaries(tokens_with_eos, eos_token_id=50256)
            report_zero = audit_eos_boundaries(tokens_without_eos, eos_token_id=50256)

        with Then("the clean sequence passes with PASS_EOS_PRESENT and has_eos=True"):
            assert report_clean["has_eos"] is True
            assert report_clean["verdict"] == "PASS_EOS_PRESENT"
            assert report_clean["eos_count"] == 3
        with And("the un-delimited sequence fails with FAIL_ZERO_EOS"):
            assert report_zero["has_eos"] is False
            assert report_zero["verdict"] == "FAIL_ZERO_EOS"


def test_ngram_contamination_clean_and_contaminated():
    with Scenario("Benchmark: N-Gram Contamination Detection"):
        with Given("clean disjoint sequences and identical contaminated sequences"):
            train_clean = list(range(100))
            eval_clean = list(range(100, 200))
            eval_contaminated = list(range(100))

        with When("evaluating 13-gram rolling hash overlaps"):
            report_clean = evaluate_ngram_contamination(train_clean, eval_clean, n=13)
            report_bad = evaluate_ngram_contamination(train_clean, eval_contaminated, n=13)

        with Then("the disjoint split reports 0% contamination and clean=True"):
            assert report_clean["clean"] is True
            assert report_clean["contamination_rate"] == 0.0
        with And("the identical split reports > 90% contamination and clean=False"):
            assert report_bad["clean"] is False
            assert report_bad["contamination_rate"] > 0.90


def test_counterfactual_margin_probing():
    with Scenario("Benchmark: Counterfactual Logit Margin Probing"):
        with Given("a loaded micro Kilat model and prompt input ids"):
            model = build_model("kilat-macro-synergy", vocab_size=32, d_model=32, n_layers=2, n_heads=2)
            input_ids = torch.randint(0, 32, (1, 16))

        with When("evaluating margin between target token 5 and foil token 10"):
            res = evaluate_counterfactual_case(model, input_ids, target_token_id=5, foil_token_id=10)
            res_identical = evaluate_counterfactual_case(model, input_ids, target_token_id=5, foil_token_id=5)

        with Then("the result contains logit margin and percentage probabilities"):
            assert "logit_margin" in res
            assert 0.0 <= res["target_prob"] <= 100.0
            assert 0.0 <= res["foil_prob"] <= 100.0
        with And("an identical target and foil yields exactly 0.0 margin"):
            assert res_identical["logit_margin"] == 0.0


def test_residual_stream_and_activation_patching():
    with Scenario("Benchmark: Residual Cache Extraction & Activation Patching"):
        with Given("a micro Kilat model, clean input ids, and length-matched corrupt ids"):
            model = build_model("kilat-macro-synergy", vocab_size=32, d_model=32, n_layers=2, n_heads=2)
            clean_ids = torch.randint(0, 32, (1, 8))
            corrupt_ids = torch.randint(0, 32, (1, 8))

        with When("extracting residual stream and evaluating layer activation patching"):
            logits, residuals = forward_with_residual_cache(model, clean_ids)
            patch_report = evaluate_activation_patching(
                model, clean_ids, corrupt_ids, target_token_id=4, foil_token_id=8
            )

        with Then("residuals contains L + 1 states matching model hidden dimension"):
            assert len(residuals) == 3  # embedding + 2 layers
            assert residuals[0].shape == (1, 8, 32)
        with And("the patching report records layer effects and recovery percentages"):
            assert len(patch_report["layers"]) == 2
            assert "recovery_pct" in patch_report["layers"][0]


def test_activation_patching_negative_mismatched_lengths():
    with Scenario("Benchmark: Activation Patching Negative Test - Mismatched Lengths"):
        with Given("clean and corrupt sequences of unequal lengths"):
            model = build_model("kilat-macro-synergy", vocab_size=32, d_model=32, n_layers=2, n_heads=2)
            clean_ids = torch.randint(0, 32, (1, 8))
            corrupt_ids = torch.randint(0, 32, (1, 16))

        with When("attempting activation patching across unequal sequence lengths"):
            with pytest.raises(ValueError) as excinfo:
                evaluate_activation_patching(model, clean_ids, corrupt_ids, target_token_id=1, foil_token_id=2)

        with Then("a ValueError is raised"):
            assert "identical shape" in str(excinfo.value)
