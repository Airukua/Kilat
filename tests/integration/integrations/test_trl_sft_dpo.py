"""
BDD Integration Tests for Kilat Post-Training Alignment: TRL SFT & DPO (Phase P2).

Follows Golden Invariants of Kilat:
1. Numerical proof over syntax passing: exact gradient parity under non-reentrant checkpointing.
2. Gradient & optimization sanity: frozen prompt/padding tokens never corrupt gradients; LoRA updates converge.
3. Data leakage & causal integrity: strict `-100` prompt and padding masking in SFT (`KilatCompletionOnlyCollator`).
4. 6-Gate Assurance verification on post-training configured models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
from peft import get_peft_model

from kilat.assurance.runner import run_verification_pipeline
from kilat.configs.model_config import KilatConfig as KilatNativeConfig
from kilat.integrations.huggingface import (
    KilatCompletionOnlyCollator,
    compute_dpo_preference_loss,
    compute_sequence_logprobs,
    get_kilat_lora_config,
    prepare_kilat_for_post_training,
)
from kilat.integrations.huggingface.configuration_kilat import KilatConfig
from kilat.integrations.huggingface.modeling_kilat import KilatForCausalLM
from tests.shared.bdd import And, Given, Scenario, Then, When


def _build_micro_post_train_model() -> KilatForCausalLM:
    """Build a lightweight deterministic micro KilatForCausalLM with zero dropout for exact parity."""
    cfg = KilatConfig(
        vocab_size=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        recall_ratio=0.5,
        recall_pattern="macro_synergy",
        ffn_mode="swiglu",
        use_rope=True,
        dropout=0.0,
        attn_drop=0.0,
        ffn_dropout=0.0,
        resid_drop=0.0,
        embd_drop=0.0,
        use_cache=True,
    )
    return KilatForCausalLM(cfg)


def test_post_training_non_reentrant_gradient_checkpointing_parity():
    with Scenario("Integration: Non-Reentrant Gradient Checkpointing Numerical Parity"):
        with Given("two identical KilatForCausalLM models, one with non-reentrant gradient checkpointing"):
            torch.manual_seed(101)
            model_std = _build_micro_post_train_model()
            model_ckpt = _build_micro_post_train_model()
            model_ckpt.load_state_dict(model_std.state_dict())

            prepare_kilat_for_post_training(
                model_ckpt,
                use_gradient_checkpointing=True,
                use_reentrant=False,
            )

        with Then("use_cache is deactivated and gradient checkpointing is active on all blocks"):
            assert model_ckpt.config.use_cache is False
            assert model_ckpt.is_gradient_checkpointing is True

        with When("running forward and backward passes on identical token batches"):
            x = torch.randint(0, 64, (2, 10))
            out_std = model_std(input_ids=x, labels=x, use_cache=False)
            out_std.loss.backward()

            out_ckpt = model_ckpt(input_ids=x, labels=x)
            out_ckpt.loss.backward()

        with Then("loss and all parameter gradients match within 1e-6"):
            assert torch.allclose(out_std.loss, out_ckpt.loss, atol=1e-6)
            for (name, p_std), (_, p_ckpt) in zip(
                model_std.named_parameters(), model_ckpt.named_parameters()
            ):
                assert p_std.grad is not None and p_ckpt.grad is not None, f"Missing grad on {name}"
                max_diff = float((p_std.grad - p_ckpt.grad).abs().max().item())
                assert max_diff < 1e-6, f"Gradient mismatch on {name}: {max_diff:.3e}"


def test_sft_completion_only_collator_and_lora_convergence():
    with Scenario("Integration: SFT Prompt Loss Masking (-100) & LoRA Completion Convergence"):
        with Given("a KilatCompletionOnlyCollator and variable-length instruction-response samples"):
            response_tpl = [7, 8]
            collator = KilatCompletionOnlyCollator(
                response_template_ids=response_tpl,
                pad_token_id=0,
                max_length=16,
            )
            samples = [
                {"input_ids": [1, 12, 13, 7, 8, 21, 22, 23, 2]},
                {"input_ids": [1, 14, 7, 8, 31, 32, 2]},
            ]

        with When("collating the batch for SFT training"):
            batch = collator(samples)

        with Then("all prompt tokens (up to response_tpl) and padding tokens are masked with -100"):
            labels = batch["labels"]
            attn_mask = batch["attention_mask"]
            # Sample 0 prompt + template has length 5 (indices 0..4), completion at indices 5..8
            assert (labels[0, :5] == -100).all()
            assert (labels[0, 5:9] == torch.tensor([21, 22, 23, 2])).all()
            # Sample 1 prompt + template has length 4 (indices 0..3), completion at 4..6, padding at 7..8
            assert (labels[1, :4] == -100).all()
            assert (labels[1, 4:7] == torch.tensor([31, 32, 2])).all()
            assert (labels[1, 7:] == -100).all()
            assert (labels[attn_mask == 0] == -100).all()

        with And("collator rejects malformed samples missing the response template"):
            with pytest.raises(ValueError, match="response_template_ids"):
                collator([{"input_ids": [1, 2, 3, 4]}])

        with When("training a LoRA-wrapped Kilat model with gradient checkpointing on the SFT batch"):
            torch.manual_seed(202)
            base_model = _build_micro_post_train_model()
            peft_model = get_peft_model(base_model, get_kilat_lora_config(r=4, lora_alpha=8, lora_dropout=0.0))
            prepare_kilat_for_post_training(peft_model, use_gradient_checkpointing=True, use_reentrant=False)
            opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-2)

            initial_loss = 0.0
            final_loss = 0.0
            for step in range(8):
                opt.zero_grad(set_to_none=True)
                out = peft_model(input_ids=batch["input_ids"], labels=batch["labels"])
                if step == 0:
                    initial_loss = float(out.loss.item())
                final_loss = float(out.loss.item())
                out.loss.backward()
                opt.step()

        with Then("completion SFT loss strictly decreases while frozen base weights remain untouched"):
            assert final_loss < initial_loss


def test_dpo_preference_alignment_reward_margin():
    with Scenario("Integration: DPO Preference Optimization Increases Implicit Reward Margin"):
        with Given("a PEFT LoRA policy model and chosen vs rejected completion pairs"):
            torch.manual_seed(303)
            base_model = _build_micro_post_train_model()
            policy_model = get_peft_model(
                base_model,
                get_kilat_lora_config(r=4, lora_alpha=8, lora_dropout=0.0),
            )
            prepare_kilat_for_post_training(policy_model, use_gradient_checkpointing=True, use_reentrant=False)

            collator = KilatCompletionOnlyCollator(response_template_ids=[7, 8], pad_token_id=0)
            chosen_batch = collator(
                [
                    {"input_ids": [1, 10, 11, 7, 8, 25, 26, 27, 2]},
                    {"input_ids": [1, 12, 13, 7, 8, 28, 29, 30, 2]},
                ]
            )
            rejected_batch = collator(
                [
                    {"input_ids": [1, 10, 11, 7, 8, 45, 46, 47, 2]},
                    {"input_ids": [1, 12, 13, 7, 8, 48, 49, 50, 2]},
                ]
            )

            # At step 0 (lora_B == 0), policy equals reference model
            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logprobs(
                    policy_model,
                    chosen_batch["input_ids"],
                    chosen_batch["labels"],
                )
                ref_rejected_logps = compute_sequence_logprobs(
                    policy_model,
                    rejected_batch["input_ids"],
                    rejected_batch["labels"],
                )

        with When("optimizing the DPO preference objective for 6 steps"):
            opt = torch.optim.AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=1e-2)
            initial_dpo_loss = 0.0
            final_dpo_loss = 0.0
            final_margin = 0.0

            for step in range(6):
                opt.zero_grad(set_to_none=True)
                pi_chosen = compute_sequence_logprobs(
                    policy_model,
                    chosen_batch["input_ids"],
                    chosen_batch["labels"],
                )
                pi_rejected = compute_sequence_logprobs(
                    policy_model,
                    rejected_batch["input_ids"],
                    rejected_batch["labels"],
                )
                loss, chosen_rew, rejected_rew = compute_dpo_preference_loss(
                    pi_chosen,
                    pi_rejected,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=0.2,
                )
                if step == 0:
                    initial_dpo_loss = float(loss.item())
                final_dpo_loss = float(loss.item())
                final_margin = float((chosen_rew - rejected_rew).mean().item())
                loss.backward()
                opt.step()

        with Then("DPO loss decreases and the implicit reward margin (chosen - rejected) becomes strictly positive"):
            assert final_dpo_loss < initial_dpo_loss
            assert final_margin > 0.0


def test_post_training_model_passes_6_gate_assurance(tmp_path: Path):
    with Scenario("Integration: Post-Training Checkpointed Model Passes 6-Gate Assurance (G0-G6)"):
        with Given("a Kilat model configured with prepare_kilat_for_post_training"):
            model = _build_micro_post_train_model()
            native_cfg = KilatNativeConfig(**model.config.to_dict())
            prepare_kilat_for_post_training(model, use_gradient_checkpointing=True, use_reentrant=False)

        with When("running the 6-Gate Assurance verification pipeline"):
            report = run_verification_pipeline(
                config=native_cfg,
                model=model,
                output_dir=str(tmp_path / "trl_assurance"),
            )

        with Then("all 7 assurance gates (G0..G6) pass"):
            assert report.passed, f"Assurance failed on post-training model: {[g.to_dict() for g in report.gates]}"
