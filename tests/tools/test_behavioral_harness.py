"""
BDD-style tests for tools/benchmarking/behavioral_harness.py (Behavioral Falsification Harness).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from tools.benchmarking.behavioral_harness import (
    evaluate_counterfactual_case,
    run_behavioral_suite,
)
from kilat import KilatConfig, KilatTransformer
from tests.shared.bdd import Scenario, Given, When, Then, And


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    p = Path("checkpoints/tinystories_gpu/checkpoint-250")
    if p.is_dir():
        return str(p)
    pytest.skip("checkpoints/tinystories_gpu/checkpoint-250 not found")


def test_behavioral_harness_single_case(checkpoint_path: str, device: torch.device):
    with Scenario("Tools: Behavioral Harness Single Counterfactual Case Evaluation"):
        with Given("a loaded Kilat model, tokenizer, and an object tracking counterfactual prompt"):
            model = KilatTransformer.from_pretrained(checkpoint_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            prompt = "Tom has a red ball and a blue ball. He gives the red ball to Anna. Tom still has a"
            expected = " blue"
            foil = " red"

            with When("evaluate_counterfactual_case computes target and foil probabilities"):
                res = evaluate_counterfactual_case(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    expected_target=expected,
                    foil_target=foil,
                    device=device,
                )

                with Then("the result contains logit margin, probabilities, and correctness boolean"):
                    assert "logit_margin" in res
                    assert "target_prob" in res
                    assert "foil_prob" in res
                    assert "is_correct" in res
                    assert isinstance(res["is_correct"], bool)

                with And("the probabilities lie within the [0, 100] percentage interval"):
                    assert 0.0 <= res["target_prob"] <= 100.0
                    assert 0.0 <= res["foil_prob"] <= 100.0


def test_behavioral_harness_suite_aggregation(checkpoint_path: str, tmp_path: Path):
    with Scenario("Tools: Behavioral Harness Suite Aggregation & Categories"):
        with Given("a custom behavioral test cases JSON with multiple prompts"):
            test_cases = [
                {
                    "category": "object_tracking",
                    "case_id": "TEST_01",
                    "prompt": "Tom has a red ball and a blue ball. He gives the red ball to Anna. Tom still has a",
                    "expected_target": " blue",
                    "foil_target": " red",
                    "description": "Test case deduction",
                },
                {
                    "category": "object_tracking",
                    "case_id": "TEST_02",
                    "prompt": "Alice has a cat and a dog. Alice lost her cat. Alice still has her",
                    "expected_target": " dog",
                    "foil_target": " cat",
                    "description": "Second test case",
                },
            ]
            tc_file = tmp_path / "custom_cases.json"
            tc_file.write_text(json.dumps(test_cases))

            with When("run_behavioral_suite executes across the test suite"):
                report = run_behavioral_suite(checkpoint_path, tc_file, device_str="auto")

                with Then("the summary accurately reflects total and passed cases"):
                    assert report["total_cases"] == 2
                    assert 0 <= report["passed_cases"] <= 2
                    assert "accuracy_pct" in report
                    assert "mean_logit_margin" in report

                with And("the category breakdown includes object_tracking with total=2"):
                    assert "object_tracking" in report["category_summary"]
                    assert report["category_summary"]["object_tracking"]["total"] == 2


def test_behavioral_harness_negative_identical_target_and_foil(checkpoint_path: str, device: torch.device):
    with Scenario("Tools: Behavioral Harness Negative Testing - Identical Target and Foil"):
        with Given("a prompt where target and foil are the exact same token"):
            model = KilatTransformer.from_pretrained(checkpoint_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)

            with When("evaluating logit margin between identical tokens"):
                res = evaluate_counterfactual_case(
                    model=model,
                    tokenizer=tokenizer,
                    prompt="The color of the sky is",
                    expected_target=" blue",
                    foil_target=" blue",
                    device=device,
                )

                with Then("the logit margin is identically 0.0"):
                    assert res["logit_margin"] == 0.0

                with And("target and foil probabilities are equal"):
                    assert res["target_prob"] == res["foil_prob"]


def test_behavioral_harness_negative_missing_test_cases(checkpoint_path: str, tmp_path: Path):
    with Scenario("Tools: Behavioral Harness Negative Testing - Missing Test Cases File"):
        with Given("a nonexistent test cases JSON path"):
            missing_file = tmp_path / "nonexistent_cases.json"

            with When("running the behavioral suite"):
                with Then("a FileNotFoundError is raised"):
                    with pytest.raises(FileNotFoundError):
                        run_behavioral_suite(checkpoint_path, missing_file)


def test_behavioral_harness_negative_missing_checkpoint(tmp_path: Path):
    with Scenario("Tools: Behavioral Harness Negative Testing - Missing Checkpoint Directory"):
        with Given("a nonexistent checkpoint directory"):
            missing_ckpt = tmp_path / "nonexistent_ckpt"
            tc_file = tmp_path / "cases.json"
            tc_file.write_text("[]")

            with When("running the behavioral suite with the missing checkpoint"):
                with Then("an exception is raised"):
                    with pytest.raises(Exception):
                        run_behavioral_suite(missing_ckpt, tc_file)
