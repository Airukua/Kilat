"""
BDD Integration Tests for Kilat PEFT LoRA Adaptation & Merging (Phase P1).

Follows Golden Invariants of Kilat:
1. Numerical proof over syntax passing.
2. Gradient and optimization sanity: only adapter parameters receive gradients.
3. Causal integrity: loss is causal cross-entropy on shifted labels.
4. Deterministic adapter saving, reloading, and merge_and_unload parity.
5. Recurrent stability: log_lambda parameters must never be targeted by LoRA.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from kilat.assurance.gates.gate1_theory import run_gate1_theory
from kilat.assurance.runner import run_verification_pipeline
from kilat.assurance.types import AssuranceError
from kilat.configs.model_config import KilatConfig as KilatNativeConfig
from kilat.integrations.huggingface.configuration_kilat import KilatConfig
from kilat.integrations.huggingface.modeling_kilat import KilatForCausalLM
from kilat.integrations.huggingface.peft import get_kilat_lora_config
from tests.shared.bdd import And, Given, Scenario, Then, When


def _build_micro_kilat_hf() -> KilatForCausalLM:
    """Build a lightweight in-memory KilatForCausalLM model (~15K parameters) to minimize RAM/CPU."""
    cfg = KilatConfig(
        vocab_size=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        recall_ratio=0.5,
        recall_pattern="macro_synergy",
        ffn_mode="swiglu",
        use_rope=True,
    )
    return KilatForCausalLM(cfg)


def test_peft_lora_model_wrapping_and_freeze():
    with Scenario("Integration: PEFT LoRA Adapter Wrapping & Parameter Isolation"):
        with Given("a KilatForCausalLM model and canonical LoRA config"):
            base_model = _build_micro_kilat_hf()
            lora_cfg = get_kilat_lora_config(r=4, lora_alpha=8)

        with When("wrapping the model with get_peft_model"):
            peft_model = get_peft_model(base_model, lora_cfg)

        with Then("trainable parameters are isolated strictly to LoRA adapters"):
            trainable_params = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
            frozen_params = [(n, p) for n, p in peft_model.named_parameters() if not p.requires_grad]

            trainable_numel = sum(p.numel() for _, p in trainable_params)
            frozen_numel = sum(p.numel() for _, p in frozen_params)

            assert len(trainable_params) > 0
            assert all("lora_" in n for n, _ in trainable_params)
            assert trainable_numel < frozen_numel

        with And("base transformer weights and recurrent decay parameters remain frozen"):
            for n, p in peft_model.named_parameters():
                if "lora_" not in n:
                    assert p.requires_grad is False


def test_peft_lora_gradient_backpropagation():
    with Scenario("Integration: PEFT LoRA Gradient & Optimization Sanity"):
        with Given("a LoRA-wrapped Kilat model and input tokens with labels"):
            base_model = _build_micro_kilat_hf()
            lora_cfg = get_kilat_lora_config(r=4, lora_alpha=8)
            peft_model = get_peft_model(base_model, lora_cfg)
            optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-2)

            input_ids = torch.randint(0, 64, (2, 8))
            labels = input_ids.clone()

        with When("running step 1 (B=0 init) followed by step 2 after lora_B update"):
            # Step 1: lora_B is zero-initialized, so lora_B gets gradients and updates
            out1 = peft_model(input_ids=input_ids, labels=labels)
            assert torch.isfinite(out1.loss), "Step 1 loss must be finite"
            out1.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Step 2: Now lora_B != 0, so both lora_A and lora_B must receive non-zero gradients
            out2 = peft_model(input_ids=input_ids, labels=labels)
            assert torch.isfinite(out2.loss), "Step 2 loss must be finite"
            out2.loss.backward()

        with Then("every trainable LoRA parameter (both lora_A and lora_B) receives a non-zero, finite gradient"):
            for name, param in peft_model.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Trainable parameter {name} has no gradient"
                    assert torch.isfinite(param.grad).all(), f"Gradient of {name} is not finite"
                    assert param.grad.abs().sum().item() > 0.0, f"Gradient of {name} is dead"

        with And("all frozen parameters have None gradient"):
            for name, param in peft_model.named_parameters():
                if not param.requires_grad:
                    assert param.grad is None, f"Frozen parameter {name} received non-None gradient"


def test_peft_lora_save_reload_and_merge(tmp_path: Path):
    with Scenario("Integration: PEFT LoRA Adapter Serialization & Weight Merging"):
        adapter_dir = tmp_path / "lora_adapter"

        with Given("a trained LoRA adapter on Kilat"):
            torch.manual_seed(42)
            base_model = _build_micro_kilat_hf()
            base_state = {k: v.clone() for k, v in base_model.state_dict().items()}
            lora_cfg = get_kilat_lora_config(r=4, lora_alpha=8, lora_dropout=0.0)
            peft_model = get_peft_model(base_model, lora_cfg)
            peft_model.eval()

            optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-2)
            x = torch.randint(0, 64, (2, 8))
            out = peft_model(input_ids=x, labels=x)
            out.loss.backward()
            optimizer.step()

        with When("saving adapter weights to disk"):
            peft_model.save_pretrained(str(adapter_dir))

        with Then("adapter configuration and safetensors weights exist"):
            assert (adapter_dir / "adapter_config.json").is_file()
            assert (
                (adapter_dir / "adapter_model.safetensors").is_file()
                or (adapter_dir / "adapter_model.bin").is_file()
            )

        with When("reloading the adapter onto a fresh base model with identical base weights"):
            fresh_base = _build_micro_kilat_hf()
            fresh_base.load_state_dict(base_state)
            reloaded_peft = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
            reloaded_peft.eval()

        with Then("reloaded model generates logits identical to the trained adapter"):
            with torch.no_grad():
                expected_logits = peft_model(input_ids=x).logits
                actual_logits = reloaded_peft(input_ids=x).logits
            assert torch.allclose(expected_logits, actual_logits, atol=1e-5)

        with When("merging LoRA weights back into the base model via merge_and_unload"):
            merged_model = reloaded_peft.merge_and_unload()
            merged_model.eval()

        with Then("the merged model is a pure KilatForCausalLM and produces identical logits"):
            assert isinstance(merged_model, KilatForCausalLM)
            with torch.no_grad():
                merged_logits = merged_model(input_ids=x).logits
            assert torch.allclose(expected_logits, merged_logits, atol=1e-4)


def test_peft_lora_rejects_forbidden_decay_targets():
    with Scenario("Integration: Invariant Guard Rejects Forbidden Recurrence Targets"):
        with Given("an operator attempting to target decay_lambda with LoRA"):
            bad_targets = ["gate", "log_lambda"]

        with When("requesting a Kilat LoRA configuration"):
            with pytest.raises(ValueError, match="Cannot apply LoRA to recurrent dynamics"):
                get_kilat_lora_config(target_modules=bad_targets)


def test_peft_lora_6_gate_assurance_pipeline(tmp_path: Path):
    with Scenario("Integration: PEFT LoRA Model Passes Full 6-Gate Assurance Pipeline (G0-G6)"):
        with Given("a LoRA-wrapped Kilat model and native KilatConfig"):
            base_model = _build_micro_kilat_hf()
            native_cfg = KilatNativeConfig(**base_model.config.to_dict())
            lora_cfg = get_kilat_lora_config(r=4, lora_alpha=8, lora_dropout=0.0)
            peft_model = get_peft_model(base_model, lora_cfg)

        with When("executing the 6-Gate Assurance verification pipeline (G0..G6) on the PeftModel"):
            report = run_verification_pipeline(
                config=native_cfg,
                model=peft_model,
                output_dir=str(tmp_path / "peft_assurance"),
            )

        with Then("all 7 verification gates (G0 Security + G1..G6) pass with zero errors"):
            assert report.passed, f"Assurance failed on PeftModel: {[g.to_dict() for g in report.gates]}"
            gate_statuses = {g.gate_id: g.status for g in report.gates}
            for expected_gate in (
                "G0_SECURITY_SUPPLY_CHAIN",
                "G1_THEORY_INIT",
                "G2_NUMERICAL_CAUSAL",
                "G3_EQUIVALENCE",
                "G4_DATA_FORENSICS",
                "G5_MICRO_DYNAMICS",
                "G6_SYSTEM_PARITY",
            ):
                assert gate_statuses.get(expected_gate) == "PASS"

        with And("Gate 1 rejects a raw LoraConfig that bypasses get_kilat_lora_config to target wte"):
            bad_raw_lora = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=4,
                lora_alpha=8,
                target_modules=["wte"],
            )
            bad_peft_model = get_peft_model(_build_micro_kilat_hf(), bad_raw_lora)
            with pytest.raises(AssuranceError, match="forbidden structural modules"):
                run_gate1_theory(native_cfg, model=bad_peft_model)
