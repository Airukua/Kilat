"""
BDD-style tests for tools/interpretability/activation_patching.py (Causal Activation Patching).

Follows Kilat BDD standards with Scenario, Given, When, Then, and And.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from tools.interpretability.activation_patching import (
    forward_with_patch,
    forward_with_residual_cache,
    get_token_id,
    run_activation_patching,
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


def test_activation_patching_get_token_id(checkpoint_path: str):
    with Scenario("Tools: Activation Patching Token ID Resolution"):
        with Given("a tokenizer loaded from checkpoint"):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)

            with When("resolving token string ' ball'"):
                tok_id = get_token_id(tokenizer, " ball")

                with Then("a valid positive integer token ID is returned within vocab bounds"):
                    assert isinstance(tok_id, int)
                    assert 0 <= tok_id < tokenizer.vocab_size


def test_activation_patching_residual_cache(checkpoint_path: str, device: torch.device):
    with Scenario("Tools: Activation Patching Residual Stream Extraction"):
        with Given("a loaded Kilat model and tokenized input ids"):
            model = KilatTransformer.from_pretrained(checkpoint_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            input_ids = tokenizer("The little boy ran into the garden", return_tensors="pt")["input_ids"].to(device)
            B, N = input_ids.shape

            with When("forward_with_residual_cache runs forward pass"):
                logits, residuals = forward_with_residual_cache(model, input_ids)

                with Then("logits have shape [B, N, vocab_size]"):
                    assert logits.shape == (B, N, model.config.vocab_size)

                with And("the residual list has length L + 1 with matching hidden dimension"):
                    assert len(residuals) == model.config.n_layer + 1
                    for act in residuals:
                        assert act.shape == (B, N, model.config.n_embd)


def test_activation_patching_forward_with_patch(checkpoint_path: str, device: torch.device):
    with Scenario("Tools: Activation Patching Forward Pass with Injected Activation"):
        with Given("model residuals from a clean run"):
            model = KilatTransformer.from_pretrained(checkpoint_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            input_ids = tokenizer("A cat jumped over the fence", return_tensors="pt")["input_ids"].to(device)
            _, residuals = forward_with_residual_cache(model, input_ids)

            with When("forward_with_patch injects layer 2 activation into layer 2 computation"):
                patched_logits = forward_with_patch(
                    model=model,
                    input_ids=input_ids,
                    patch_layer=2,
                    patch_tensor=residuals[3],
                    patch_pos=-1,
                )

                with Then("the patched output logits have shape [B, N, vocab_size]"):
                    assert patched_logits.shape == (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)


def test_activation_patching_end_to_end(checkpoint_path: str):
    with Scenario("Tools: Activation Patching End-to-End Layer-by-Layer Interventions"):
        with Given("length-matched clean and corrupt prompts with counterfactual targets"):
            clean_prompt = "Tom has a red ball and a blue ball. He gives the red ball to Anna. Tom still has a"
            corrupt_prompt = "Tom has a blue ball and a red ball. He gives the blue ball to Anna. Tom still has a"
            target = " blue"
            foil = " red"

            with When("run_activation_patching evaluates recovery percentage across layers"):
                report = run_activation_patching(
                    model_dir=checkpoint_path,
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    target_token=target,
                    foil_token=foil,
                    device_str="auto",
                )

                with Then("the report records clean and corrupt logit diffs and layer effects"):
                    assert "clean_logit_diff" in report
                    assert "corrupt_logit_diff" in report
                    assert "total_logit_diff_span" in report
                    assert "layer_effects" in report
                    assert len(report["layer_effects"]) > 0

                with And("each layer entry contains recovery percentage and target logit"):
                    for eff in report["layer_effects"]:
                        assert "layer" in eff
                        assert "recovery_pct" in eff
                        assert "logit_diff" in eff
                        assert "target_logit" in eff


def test_activation_patching_negative_mismatched_prompt_lengths(checkpoint_path: str):
    with Scenario("Tools: Activation Patching Negative Testing - Mismatched Prompt Lengths"):
        with Given("prompts with different token sequence lengths"):
            clean_prompt = "Short prompt"
            corrupt_prompt = "This is a significantly longer prompt with extra words"

            with When("attempting activation patching across unequal prompt lengths"):
                with Then("a ValueError is raised"):
                    with pytest.raises(ValueError, match="Prompt lengths must match"):
                        run_activation_patching(
                            model_dir=checkpoint_path,
                            clean_prompt=clean_prompt,
                            corrupt_prompt=corrupt_prompt,
                            target_token=" blue",
                            foil_token=" red",
                        )


def test_activation_patching_negative_empty_token_string(checkpoint_path: str):
    with Scenario("Tools: Activation Patching Negative Testing - Empty Token String"):
        with Given("an empty token string ''"):
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)

            with When("resolving token ID for empty string"):
                with Then("a ValueError is raised"):
                    with pytest.raises(ValueError, match="Could not encode token string"):
                        get_token_id(tokenizer, "")


def test_activation_patching_negative_missing_checkpoint(tmp_path: Path):
    with Scenario("Tools: Activation Patching Negative Testing - Missing Checkpoint Directory"):
        with Given("a nonexistent checkpoint directory"):
            missing = tmp_path / "nonexistent_model"

            with When("running activation patching"):
                with Then("an exception is raised"):
                    with pytest.raises(Exception):
                        run_activation_patching(
                            model_dir=missing,
                            clean_prompt="Hello world",
                            corrupt_prompt="Hello world",
                            target_token=" world",
                            foil_token=" dog",
                        )
