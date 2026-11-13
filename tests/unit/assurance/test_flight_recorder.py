"""
Unit and regression tests for FlightRecorderCallback.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kilat.assurance.flight_recorder import FlightRecorderCallback
from kilat.engine.callbacks import TrainerControl, TrainerState
from tests.shared.bdd import And, Given, Scenario, Then, When

_METRIC_TRAIN_CE_LOSS = "train/ce_loss"


def test_flight_recorder_eval_log_isolation_regression(tmp_path: Path) -> None:
    with Scenario("Regression: FlightRecorderCallback isolates evaluation metrics from training telemetry"):
        with Given("a FlightRecorderCallback with output directory and accumulated training tokens"):
            recorder = FlightRecorderCallback(output_dir=str(tmp_path))
            recorder._tokens_since_log = 512
            state = TrainerState(global_step=1000)
            control = TrainerControl()

        with When("on_log is dispatched with an evaluation metrics payload"):
            eval_payload = {
                "eval_loss": 1.60,
                "eval_ppl": 4.95,
                "eval_aux_loss": 0.02,
            }
            recorder.on_log(None, state, control, logs=eval_payload)

        with Then("the evaluation logs dictionary is not polluted with fake train metrics"):
            assert _METRIC_TRAIN_CE_LOSS not in eval_payload
            assert "train/ppl" not in eval_payload
            assert "train/aux_loss" not in eval_payload
            assert eval_payload["eval_loss"] == 1.60

        with And("telemetry_history records genuine evaluation metrics without fake zero-loss"):
            assert len(recorder.telemetry_history) == 1
            entry = recorder.telemetry_history[0]
            assert entry["step"] == 1000.0
            assert entry["eval/loss"] == 1.60
            assert entry["eval/ppl"] == 4.95
            assert entry["eval/aux_loss"] == 0.02
            assert _METRIC_TRAIN_CE_LOSS not in entry
            assert "train/ppl" not in entry

        with And("accumulated training tokens are preserved for the subsequent training step"):
            assert recorder._tokens_since_log == 512

        with And("telemetry.jsonl contains the clean evaluation entry"):
            telemetry_file = tmp_path / "telemetry.jsonl"
            assert telemetry_file.is_file()
            lines = [json.loads(line) for line in telemetry_file.read_text().strip().split("\n")]
            assert len(lines) == 1
            assert lines[0]["eval/loss"] == 1.60
            assert lines[0]["eval/ppl"] == 4.95
            assert _METRIC_TRAIN_CE_LOSS not in lines[0]

        with When("subsequent training log is dispatched with training loss"):
            train_payload = {
                "loss": 2.15,
                "ce_loss": 2.15,
                "aux_loss": 0.01,
            }
            state.global_step = 1001
            recorder.on_log(None, state, control, logs=train_payload)

        with Then("training metrics are processed normally and token counter is consumed"):
            assert len(recorder.telemetry_history) == 2
            train_entry = recorder.telemetry_history[1]
            assert train_entry["step"] == 1001.0
            assert train_entry[_METRIC_TRAIN_CE_LOSS] == 2.15
            assert recorder._tokens_since_log == 0
