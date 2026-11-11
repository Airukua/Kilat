from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Sequence
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from kilat.assurance.chaos.saboteur import Saboteur, SimulatedPreemptionError
from kilat.assurance.gates import (
    run_gate2_numerical,
    run_gate3_equivalence,
    run_gate4_data,
    run_gate5_dynamics,
)
from kilat.assurance.runner import run_verification_pipeline
from kilat.assurance.types import AssuranceError, AssuranceFailureCode
from kilat.configs.model_config import KilatConfig
from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import SyntheticTokenDataset
from kilat.engine.callbacks import CallbackHandler, TrainerState
from kilat.engine.checkpoint import load_checkpoint, save_checkpoint
from kilat.engine.optimizer import create_optimizer
from kilat.engine.scheduler import get_scheduler
from kilat.models.transformer import KilatTransformer


@dataclass
class ChaosScenarioResult:
    scenario_id: str
    name: str
    caught: bool
    expected_codes: list[str]
    observed_code: str
    duration_ms: float
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "caught": self.caught,
            "expected_codes": self.expected_codes,
            "observed_code": self.observed_code,
            "duration_ms": round(self.duration_ms, 3),
            "details": self.details,
        }


@dataclass
class ChaosSuiteReport:
    baseline_passed: bool
    post_clean_passed: bool
    scenarios: list[ChaosScenarioResult]

    @property
    def scenarios_total(self) -> int:
        return len(self.scenarios)

    @property
    def scenarios_caught(self) -> int:
        return sum(1 for s in self.scenarios if s.caught)

    @property
    def false_negatives(self) -> int:
        return self.scenarios_total - self.scenarios_caught

    @property
    def passed(self) -> bool:
        return self.baseline_passed and self.post_clean_passed and self.false_negatives == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": True,
            "baseline_passed": self.baseline_passed,
            "post_clean_passed": self.post_clean_passed,
            "scenarios_total": self.scenarios_total,
            "scenarios_caught": self.scenarios_caught,
            "false_negatives": self.false_negatives,
            "overall_status": "PASS" if self.passed else "FAIL",
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


ALL_SCENARIO_IDS = ("C-01", "C-02", "C-03", "C-04", "C-05", "C-06", "C-07", "C-08")


def run_chaos_suite(scenarios: str | Sequence[str] = "all") -> ChaosSuiteReport:
    """
    Execute Layer-2 Active Chaos Engineering Suite (`kilat chaos`).

    1. Runs clean baseline verification (must pass 100% with 0 false positives).
    2. Injects each requested scenario (`C-01` .. `C-08`) strictly in RAM / `tmp_path`
       and asserts that the corresponding Assurance Gate catches it with the exact
       expected `AssuranceFailureCode` (0% false negatives).
    3. Re-verifies the model after exiting all `with saboteur.*:` blocks to prove
       100% non-destructive hook cleanup.
    """
    if isinstance(scenarios, str):
        if scenarios.strip().lower() == "all":
            selected = list(ALL_SCENARIO_IDS)
        else:
            selected = [s.strip().upper() for s in scenarios.split(",") if s.strip()]
    else:
        selected = [s.strip().upper() for s in scenarios]

    unknown = [s for s in selected if s not in ALL_SCENARIO_IDS]
    if unknown:
        raise ValueError(f"Unknown chaos scenario IDs: {unknown}. Valid IDs: {ALL_SCENARIO_IDS}")

    cfg = KilatConfig(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        n_layer=2,
        max_position_embeddings=32,
        dropout=0.0,
        ffn_mode="moe",
        num_experts=4,
        active_experts=1,
        num_shared_experts=1,
        aux_loss_coef=0.01,
        use_rope=True,
        rope_head_dim=4,
    )
    model = KilatTransformer(cfg)
    collator = KilatDataCollator(pad_token_id=0, max_length=16)
    train_ds = SyntheticTokenDataset(vocab_size=128, seq_len=16, num_samples=16, seed=301)
    eval_ds = SyntheticTokenDataset(vocab_size=128, seq_len=16, num_samples=8, seed=302)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collator)
    eval_dl = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collator)

    # Step 1: Clean baseline check
    baseline_report = run_verification_pipeline(
        config=cfg,
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        full=False,
    )
    baseline_passed = baseline_report.passed

    saboteur = Saboteur()
    results: list[ChaosScenarioResult] = []

    for sc_id in selected:
        t0 = time.perf_counter()

        if sc_id == "C-01":
            expected = [AssuranceFailureCode.ERR_MISSING_PARAMETER_GRAD.value]
            caught = False
            observed = "NONE"
            detail = ""
            with saboteur.detach_global_decay_branch(model):
                try:
                    run_gate2_numerical(cfg, model=model, skip_gradcheck=True)
                except AssuranceError as exc:
                    observed = exc.code.value
                    caught = observed in expected
                    detail = exc.message
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-01",
                    name="Silent Autograd Branch Detachment",
                    caught=caught,
                    expected_codes=expected,
                    observed_code=observed,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=detail,
                )
            )

        elif sc_id == "C-02":
            expected = [AssuranceFailureCode.ERR_CAUSAL_FUTURE_LEAK.value]
            caught = False
            observed = "NONE"
            detail = ""
            with saboteur.inject_causal_future_leak(model, leak_scale=1e-3):
                try:
                    run_gate2_numerical(cfg, model=model, skip_gradcheck=True)
                except AssuranceError as exc:
                    observed = exc.code.value
                    caught = observed in expected
                    detail = exc.message
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-02",
                    name="Subtle Causal Future Leak",
                    caught=caught,
                    expected_codes=expected,
                    observed_code=observed,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=detail,
                )
            )

        elif sc_id == "C-03":
            expected = [AssuranceFailureCode.ERR_BATCH_ISOLATION_BREACH.value]
            caught = False
            observed = "NONE"
            detail = ""
            with saboteur.corrupt_batch_isolation(model):
                try:
                    run_gate2_numerical(cfg, model=model, skip_gradcheck=True)
                except AssuranceError as exc:
                    observed = exc.code.value
                    caught = observed in expected
                    detail = exc.message
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-03",
                    name="Cross-Batch Tensor Reshape Mix",
                    caught=caught,
                    expected_codes=expected,
                    observed_code=observed,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=detail,
                )
            )

        elif sc_id == "C-04":
            expected = [AssuranceFailureCode.ERR_NON_FINITE_ACTIVATION.value]
            caught = False
            observed = "NONE"
            detail = ""
            with saboteur.inject_fp16_rmsnorm_overflow(model):
                try:
                    run_gate5_dynamics(cfg, model=model, memorization_steps=5)
                except AssuranceError as exc:
                    observed = exc.code.value
                    caught = observed in expected
                    detail = exc.message
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-04",
                    name="FP16 >256 Square Overflow & Inf Grad Storm",
                    caught=caught,
                    expected_codes=expected,
                    observed_code=observed,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=detail,
                )
            )

        elif sc_id == "C-05":
            expected = [AssuranceFailureCode.ERR_DECAY_NORMALIZATION_DIVERGENCE.value]
            caught = False
            observed = "NONE"
            detail = ""
            with saboteur.inject_naive_decay_singularity(model):
                try:
                    run_gate3_equivalence(cfg, model=model)
                except AssuranceError as exc:
                    observed = exc.code.value
                    caught = observed in expected
                    detail = exc.message
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-05",
                    name="Decay Horizon Singularity (lambda -> 1^-)",
                    caught=caught,
                    expected_codes=expected,
                    observed_code=observed,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=detail,
                )
            )

        elif sc_id == "C-06":
            expected = [
                AssuranceFailureCode.ERR_ROUTER_COLLAPSE.value,
                AssuranceFailureCode.ERR_ZERO_PARAMETER_GRAD.value,
            ]
            caught_collapse = False
            caught_zero_grad = False
            obs_codes: list[str] = []

            with saboteur.collapse_moe_router(model):
                try:
                    run_gate5_dynamics(cfg, model=model, memorization_steps=5)
                except AssuranceError as exc:
                    obs_codes.append(exc.code.value)
                    caught_collapse = exc.code == AssuranceFailureCode.ERR_ROUTER_COLLAPSE

            with saboteur.kill_top1_router_grad(model):
                try:
                    run_gate2_numerical(cfg, model=model, skip_gradcheck=True)
                except AssuranceError as exc:
                    obs_codes.append(exc.code.value)
                    caught_zero_grad = exc.code == AssuranceFailureCode.ERR_ZERO_PARAMETER_GRAD

            both_caught = caught_collapse and caught_zero_grad
            results.append(
                ChaosScenarioResult(
                    scenario_id="C-06",
                    name="MoE Router Collapse & Top-1 Zero-Grad",
                    caught=both_caught,
                    expected_codes=expected,
                    observed_code="+".join(obs_codes) if obs_codes else "NONE",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=f"collapse_caught={caught_collapse}, top1_zero_grad_caught={caught_zero_grad}",
                )
            )

        elif sc_id == "C-07":
            expected = [
                AssuranceFailureCode.ERR_EMPTY_TARGET_BATCH.value,
                AssuranceFailureCode.ERR_TOKEN_OUT_OF_BOUNDS.value,
            ]
            caught_empty = False
            caught_oov = False
            obs_codes_7: list[str] = []

            with saboteur.poison_dataloader_batches(train_dl, mode="all_masked", vocab_size=cfg.vocab_size) as bad_dl1:
                # Also prove M-06 model loss guard produces finite 0.0 instead of NaN!
                with torch.no_grad():
                    loss_m06 = model(bad_dl1[0]["input_ids"], labels=bad_dl1[0]["labels"], return_dict=True).loss
                m06_finite = loss_m06 is not None and bool(torch.isfinite(loss_m06).all().item())
                try:
                    run_gate4_data(bad_dl1, vocab_size=cfg.vocab_size)
                except AssuranceError as exc:
                    obs_codes_7.append(exc.code.value)
                    caught_empty = (exc.code == AssuranceFailureCode.ERR_EMPTY_TARGET_BATCH) and m06_finite

            with saboteur.poison_dataloader_batches(train_dl, mode="oov_token", vocab_size=cfg.vocab_size) as bad_dl2:
                try:
                    run_gate4_data(bad_dl2, vocab_size=cfg.vocab_size)
                except AssuranceError as exc:
                    obs_codes_7.append(exc.code.value)
                    caught_oov = exc.code == AssuranceFailureCode.ERR_TOKEN_OUT_OF_BOUNDS

            results.append(
                ChaosScenarioResult(
                    scenario_id="C-07",
                    name="Poisoned Batch (All -100 & OOV Token)",
                    caught=(caught_empty and caught_oov),
                    expected_codes=expected,
                    observed_code="+".join(obs_codes_7) if obs_codes_7 else "NONE",
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=f"empty_target_caught={caught_empty}, oov_caught={caught_oov}",
                )
            )

        elif sc_id == "C-08":
            expected = [AssuranceFailureCode.ERR_CHECKPOINT_CHECKSUM_MISMATCH.value]
            preemption_safe = False
            checksum_caught = False
            observed_8 = "NONE"

            with tempfile.TemporaryDirectory() as tmp_ckpt_root:
                opt = create_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
                sched = get_scheduler("linear", opt, num_warmup_steps=1, num_training_steps=5)
                st = TrainerState(global_step=2, max_steps=5, is_world_process_zero=True)
                cb = CallbackHandler([])

                # Save healthy checkpoint first
                save_checkpoint(
                    model=model,
                    optimizer=opt,
                    scheduler=sched,
                    scaler=None,
                    state=st,
                    callback_handler=cb,
                    output_dir=tmp_ckpt_root,
                    tag="latest",
                    atomic=True,
                )
                ckpt_path = os.path.join(tmp_ckpt_root, "checkpoint-latest")
                if not (os.path.isdir(ckpt_path)):
                    raise RuntimeError("Assurance invariant violated (survives python -O): os.path.isdir(ckpt_path)")
                # Sub-test A: Mid-save crash inside write_checkpoint_manifest leaves healthy checkpoint intact
                # and cleans up temporary `.tmp_checkpoint_*` directory!
                with patch(
                    "kilat.engine.checkpoint.write_checkpoint_manifest",
                    side_effect=SimulatedPreemptionError("SIGUSR1 preemption mid-save"),
                ):
                    try:
                        save_checkpoint(
                            model=model,
                            optimizer=opt,
                            scheduler=sched,
                            scaler=None,
                            state=st,
                            callback_handler=cb,
                            output_dir=tmp_ckpt_root,
                            tag="latest",
                            atomic=True,
                        )
                    except SimulatedPreemptionError:
                        pass

                leftover_tmps = [d for d in os.listdir(tmp_ckpt_root) if d.startswith(".tmp_checkpoint_")]
                preemption_safe = (len(leftover_tmps) == 0) and os.path.isdir(ckpt_path)

                # Sub-test B: 1-byte corruption caught by SHA-256 manifest verification on load_checkpoint
                with saboteur.corrupt_checkpoint_byte(ckpt_path):
                    try:
                        load_checkpoint(
                            model=model,
                            optimizer=opt,
                            scheduler=sched,
                            scaler=None,
                            state=st,
                            callback_handler=cb,
                            checkpoint_path=ckpt_path,
                            device=torch.device("cpu"),
                        )
                    except (RuntimeError, ValueError) as exc:
                        if AssuranceFailureCode.ERR_CHECKPOINT_CHECKSUM_MISMATCH.value in str(exc):
                            checksum_caught = True
                            observed_8 = AssuranceFailureCode.ERR_CHECKPOINT_CHECKSUM_MISMATCH.value

            results.append(
                ChaosScenarioResult(
                    scenario_id="C-08",
                    name="Mid-Checkpoint Crash & Corrupted State SHA-256 Guard",
                    caught=(preemption_safe and checksum_caught),
                    expected_codes=expected,
                    observed_code=observed_8,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                    details=f"preemption_atomic_safe={preemption_safe}, checksum_caught={checksum_caught}",
                )
            )

    # Step 3: Post-chaos clean re-verification proving zero residual state/hook leakage
    post_report = run_verification_pipeline(
        config=cfg,
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        full=False,
    )
    post_clean_passed = post_report.passed

    return ChaosSuiteReport(
        baseline_passed=baseline_passed,
        post_clean_passed=post_clean_passed,
        scenarios=results,
    )
