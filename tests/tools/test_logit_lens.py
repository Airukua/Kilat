"""
BDD-style tests for tools/interpretability/logit_lens.py (Kilat Logit Lens Inspector).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from tools.interpretability.logit_lens import run_logit_lens
from tests.shared.bdd import Scenario, Given, When, Then, And


@pytest.fixture(scope="module")
def checkpoint_path() -> str:
    p = Path("checkpoints/tinystories_gpu/checkpoint-250")
    if p.is_dir():
        return str(p)
    pytest.skip("checkpoints/tinystories_gpu/checkpoint-250 not found")


def test_logit_lens_execution(checkpoint_path: str):
    with Scenario("Tools: Logit Lens Multi-Layer Hidden State Inspection"):
        with Given("a trained Kilat checkpoint and input prompt"):
            prompt = "Once upon a time there was a little"
            top_k = 3

            with When("run_logit_lens inspects intermediate representations across blocks"):
                res = run_logit_lens(
                    model_dir=checkpoint_path,
                    prompt=prompt,
                    top_k=top_k,
                    device_str="auto",
                )

                with Then("the analyzed layer count matches the list of layer reports"):
                    assert "layers_analyzed" in res
                    assert res["layers_analyzed"] > 0
                    assert len(res["layers"]) == res["layers_analyzed"]

                with And("each layer contains non-negative Shannon entropy and top-k predictions"):
                    for l_info in res["layers"]:
                        assert l_info["entropy"] >= 0.0
                        assert len(l_info["top_predictions"]) == top_k
                        prob_sum = sum(p["probability"] for p in l_info["top_predictions"])
                        assert prob_sum <= 100.01


def test_logit_lens_negative_missing_checkpoint(tmp_path: Path):
    with Scenario("Tools: Logit Lens Negative Testing - Missing Checkpoint Directory"):
        with Given("a nonexistent checkpoint directory"):
            missing = tmp_path / "missing_model_checkpoint"

            with When("attempting to run the logit lens"):
                with Then("an exception is raised"):
                    with pytest.raises(Exception):
                        run_logit_lens(missing, "Hello world")


def test_logit_lens_negative_variable_top_k(checkpoint_path: str):
    with Scenario("Tools: Logit Lens Negative Testing - Variable Top-K Constraints"):
        with Given("different top_k constraints k=1 and k=5"):
            with When("probing with top_k=1"):
                res1 = run_logit_lens(checkpoint_path, "A tiny dog", top_k=1)

                with Then("exactly 1 prediction item is returned per layer"):
                    assert len(res1["layers"][0]["top_predictions"]) == 1

            with When("probing with top_k=5"):
                res5 = run_logit_lens(checkpoint_path, "A tiny dog", top_k=5)

                with Then("exactly 5 prediction items are returned per layer"):
                    assert len(res5["layers"][0]["top_predictions"]) == 5
