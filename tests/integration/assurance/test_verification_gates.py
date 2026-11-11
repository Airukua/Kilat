from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from kilat.models.transformer import KilatTransformer
from kilat.assurance import (
    AssuranceError,
    AssuranceFailureCode,
    FlightRecorderCallback,
    TrainingDivergedError,
    run_gate4_data,
    run_verification_pipeline,
)
from kilat.configs.dataloader_config import DataLoaderConfig
from kilat.configs.main_config import MainConfig
from kilat.configs.model_config import KilatConfig
from kilat.configs.tokenizer_config import TokenizerConfig
from kilat.configs.training_config import TrainingConfig
from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import SyntheticTokenDataset
from kilat.engine.args import TrainingArguments
from kilat.engine.trainer import KilatTrainer
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_full_assurance_pipeline_dense_and_scaling_readiness(tmp_path: Path) -> None:
    with Scenario("Integration: Full 8-Stage Pre-Training Assurance Pipeline (Gates 1-6 + Stage 7-8)"):
        with Given("an offline MainConfig with hybrid MLA/Decay RoPE transformer and synthetic dataset"):
            main_cfg = MainConfig(
                model=KilatConfig(
                    vocab_size=128,
                    n_embd=32,
                    n_head=4,
                    n_layer=2,
                    block_size=32,
                    dropout=0.0,
                    use_rope=True,
                    rope_head_dim=4,
                ),
                training=TrainingConfig(
                    output_dir=str(tmp_path / "run_out"),
                    training_mode="steps",
                    max_steps=10,
                    learning_rate=3e-4,
                    weight_decay=0.05,
                    precision="fp32",
                ),
                dataloader=DataLoaderConfig(
                    dataset_format="synthetic",
                    max_seq_length=16,
                    train_batch_size=2,
                    eval_batch_size=2,
                    num_workers=0,
                    synthetic_num_samples=20,
                    synthetic_eval_samples=8,
                ),
                tokenizer=TokenizerConfig(
                    tokenizer_type="custom",
                    custom_builder="kilat.data.tokenizer:build_byte_tokenizer",
                    custom_kwargs={"vocab_size": 128},
                ),
            )
            cert_dir = tmp_path / "cert"

        with When("run_verification_pipeline is executed with full=True"):
            report = run_verification_pipeline(
                config=main_cfg,
                output_dir=str(cert_dir),
                full=True,
            )

        with Then("all 9 stages (G0 security + 8 numerical) pass with overall_status == 'PASS'"):
            assert report.passed, f"Assurance failed: {[g.to_dict() for g in report.gates if g.status != 'PASS']}"
            gate_ids = [g.gate_id for g in report.gates]
            assert gate_ids == [
                "G0_SECURITY_SUPPLY_CHAIN",
                "G1_THEORY_INIT",
                "G2_NUMERICAL_CAUSAL",
                "G3_EQUIVALENCE",
                "G4_DATA_FORENSICS",
                "G5_MICRO_DYNAMICS",
                "G6_SYSTEM_PARITY",
                "S7_SCALING_LAWS",
                "S8_READINESS_SIGNOFF",
            ]

        with And("signed assurance_report.json and assurance_report.md artifacts are written"):
            json_file = cert_dir / "assurance_report.json"
            md_file = cert_dir / "assurance_report.md"
            assert json_file.is_file()
            assert md_file.is_file()
            parsed = json.loads(json_file.read_text(encoding="utf-8"))
            assert parsed["overall_status"] == "PASS"
            assert len(parsed["gates"]) == 9
            assert "Kilat Pre-Training Assurance Certificate" in md_file.read_text(encoding="utf-8")


def test_assurance_pipeline_hybrid_deepseek_moe() -> None:
    with Scenario("Integration: Pre-Flight Gates 1-6 on Hybrid DeepSeekMoE Architecture"):
        with Given("a segmented DeepSeekMoE KilatConfig (fine_grained_factor=2, num_shared_experts=1)"):
            moe_cfg = KilatConfig(
                vocab_size=128,
                n_embd=32,
                n_head=4,
                n_layer=2,
                block_size=32,
                dropout=0.0,
                ffn_mode="moe",
                num_experts=4,
                active_experts=1,
                num_shared_experts=1,
                fine_grained_factor=2,
                aux_loss_coef=0.01,
                use_rope=True,
                rope_head_dim=4,
            )

        with When("run_verification_pipeline runs Gates 1 through 6"):
            report = run_verification_pipeline(config=moe_cfg, full=False)

        with Then("every MoE mathematical, causal, router-entropy, and checkpoint parity gate passes"):
            assert report.passed, f"MoE assurance failed: {[g.to_dict() for g in report.gates if g.status != 'PASS']}"
            g5 = next(g for g in report.gates if g.gate_id == "G5_MICRO_DYNAMICS")
            assert g5.metrics["min_router_entropy_norm"] > 0.30
            assert g5.metrics["final_overfit_ce"] < 0.10


def test_gate4_detects_train_val_13gram_contamination() -> None:
    with Scenario("Integration: Gate 4 Catches Exact 13-Gram Train-Validation Contamination"):
        with Given("an eval split sharing an identical 16-token sequence with the train split"):
            collator = KilatDataCollator(pad_token_id=0, max_length=16)
            train_ds = SyntheticTokenDataset(vocab_size=128, seq_len=16, num_samples=8, seed=555)
            # Identical seed => identical 13-grams leaked into validation!
            leaked_eval_ds = SyntheticTokenDataset(vocab_size=128, seq_len=16, num_samples=4, seed=555)
            train_dl = DataLoader(train_ds, batch_size=2, collate_fn=collator)
            eval_dl = DataLoader(leaked_eval_ds, batch_size=2, collate_fn=collator)

        with When("run_gate4_data audits both dataloaders"):
            with pytest.raises(AssuranceError) as exc_info:
                run_gate4_data(train_dl, vocab_size=128, eval_dataloader=eval_dl)

        with Then("Gate 4 aborts with ERR_TRAIN_VAL_CONTAMINATION"):
            assert exc_info.value.code == AssuranceFailureCode.ERR_TRAIN_VAL_CONTAMINATION


def test_flight_recorder_telemetry_and_circuit_breaker_forensics(tmp_path: Path) -> None:
    with Scenario("Integration: FlightRecorderCallback Async Telemetry & Crash Forensic Isolation"):
        with Given("a Hybrid MoE trainer with FlightRecorderCallback and an existing healthy checkpoint-best"):
            cfg = KilatConfig(
                vocab_size=64,
                n_embd=32,
                n_head=4,
                n_layer=2,
                block_size=16,
                dropout=0.0,
                ffn_mode="moe",
                num_experts=4,
                active_experts=2,
                num_shared_experts=1,
            )
            model = KilatTransformer(cfg)
            ds = SyntheticTokenDataset(vocab_size=64, seq_len=16, num_samples=16, seed=42)
            dl = DataLoader(ds, batch_size=2, collate_fn=KilatDataCollator(pad_token_id=0, max_length=16))

            out_dir = tmp_path / "flight_run"
            best_dir = out_dir / "checkpoint-best"
            best_dir.mkdir(parents=True, exist_ok=True)
            sentinel = best_dir / "healthy_sentinel.txt"
            sentinel.write_text("DO_NOT_OVERWRITE", encoding="utf-8")

            recorder = FlightRecorderCallback(max_consecutive_skipped_steps=3)
            args = TrainingArguments(
                output_dir=str(out_dir),
                training_mode="steps",
                max_steps=4,
                logging_steps=1,
                save_checkpoints=False,
                max_grad_norm=1.0,
            )
            trainer = KilatTrainer(model, args, dl, callbacks=[recorder])

        with When("4 healthy training steps execute"):
            trainer.train()

        with Then("FlightRecorder records MoE router entropy, lambda horizon, and update-to-weight ratios"):
            assert len(recorder.telemetry_history) >= 4
            latest = recorder.telemetry_history[-1]
            assert "train/ce_loss" in latest
            assert "moe/router_entropy_norm" in latest
            assert "attn/lambda_min" in latest
            assert "optim/update_to_weight_ratio" in latest

        with When("a NaN loss anomaly occurs on a subsequent run"):
            recorder_crash = FlightRecorderCallback()
            args_crash = TrainingArguments(
                output_dir=str(out_dir),
                training_mode="steps",
                max_steps=3,
                logging_steps=1,
                save_checkpoints=False,
            )

            def _nan_loss_fn(m: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                out = m(batch["input_ids"], return_dict=True)
                return out.logits.sum() * float("nan")

            trainer_crash = KilatTrainer(
                KilatTransformer(cfg),
                args_crash,
                dl,
                compute_loss_fn=_nan_loss_fn,
                callbacks=[recorder_crash],
            )
            with pytest.raises(TrainingDivergedError) as crash_exc:
                trainer_crash.train()

        with Then("an isolated forensics/crash_step_1.pt bundle is saved while checkpoint-best remains intact"):
            assert sentinel.read_text(encoding="utf-8") == "DO_NOT_OVERWRITE"
            forensic_path = crash_exc.value.forensic_path
            assert forensic_path is not None and os.path.isfile(forensic_path)
            bundle = torch.load(forensic_path, map_location="cpu", weights_only=False)
            assert bundle["step"] == 1
            assert "trigger_batch" in bundle and "input_ids" in bundle["trigger_batch"]
            assert "layer_norms" in bundle and len(bundle["layer_norms"]) > 0
