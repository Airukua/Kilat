"""
Unit tests for reporting and logging integration callbacks.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import sys
from unittest.mock import MagicMock
import pytest

from kilat.engine.args import TrainingArguments
from kilat.engine.callbacks import TrainerControl, TrainerState
from kilat.engine.integrations import (
    CometMLCallback,
    ConsoleCallback,
    IntegrationCallback,
    MLflowCallback,
    TensorBoardCallback,
    WandbCallback,
    _is_package_available,
    _sanitize_key,
    get_reporting_integration_callbacks,
    register_integration,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_integration_helpers_and_console():
    with Scenario("Unit: Integration Helpers and ConsoleCallback Lifecycle"):
        with Given("metric keys containing special characters"):
            key = "eval/loss.loss:0"
            with When("sanitizing the key"):
                sanitized = _sanitize_key(key)
                with Then("slashes, colons and special symbols are converted"):
                    assert "/" not in sanitized
                    assert ":" not in sanitized

        with Given("a ConsoleCallback instance and training state"):
            console = ConsoleCallback()
            args = TrainingArguments(output_dir="tmp_test_dir", precision="fp32")
            state = TrainerState(global_step=1, epoch=0.5)
            control = TrainerControl()

            with When("executing lifecycle hooks"):
                console.on_train_begin(args, state, control)
                console.on_log(args, state, control, logs={"loss": 2.1, "learning_rate": 0.001})
                console.on_evaluate(args, state, control, metrics={"eval_loss": 1.9})
                console.on_train_end(args, state, control)
                with Then("all console hooks execute without errors"):
                    assert True


def test_tensorboard_callback_with_mock_writer():
    with Scenario("Unit: TensorBoardCallback Scalar Logging"):
        with Given("a TensorBoardCallback initialized with a mock SummaryWriter"):
            mock_writer = MagicMock()
            tb_cb = TensorBoardCallback(tb_writer=mock_writer)
            args = TrainingArguments(output_dir="tmp_tb_dir", precision="fp32")
            state = TrainerState(global_step=10, epoch=1.0)
            control = TrainerControl()

            with When("logging train metrics and closing"):
                tb_cb.on_train_begin(args, state, control)
                tb_cb.on_log(args, state, control, logs={"loss": 0.5, "lr": 1e-4})
                tb_cb.on_train_end(args, state, control)

                with Then("the writer receives scalar log events"):
                    assert mock_writer.add_scalar.called
                    assert mock_writer.flush.called


def test_wandb_and_mlflow_mocked_integrations():
    with Scenario("Unit: Wandb and MLflow Mocked Lifecycle"):
        with Given("mocked wandb and mlflow modules in sys.modules"):
            mock_wandb = MagicMock()
            mock_mlflow = MagicMock()
            sys.modules["wandb"] = mock_wandb
            sys.modules["mlflow"] = mock_mlflow

            args = TrainingArguments(output_dir="tmp_mock_dir", precision="fp32")
            state = TrainerState(global_step=5, epoch=0.5)
            control = TrainerControl()

            with When("executing WandbCallback lifecycle"):
                wb_cb = WandbCallback()
                wb_cb.on_train_begin(args, state, control)
                wb_cb.on_log(args, state, control, logs={"train_loss": 0.7})
                wb_cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.6})
                wb_cb.on_train_end(args, state, control)
                with Then("wandb log is called"):
                    assert mock_wandb.log.called

            with When("executing MLflowCallback lifecycle"):
                mlf_cb = MLflowCallback()
                mlf_cb.on_train_begin(args, state, control)
                mlf_cb.on_log(args, state, control, logs={"train_loss": 0.8})
                mlf_cb.on_train_end(args, state, control)
                with Then("mlflow log_metrics is called"):
                    assert mock_mlflow.log_metrics.called

            with When("executing CometMLCallback with mock experiment"):
                mock_exp = MagicMock()
                mock_comet = MagicMock()
                sys.modules["comet_ml"] = mock_comet
                comet_cb = CometMLCallback(experiment=mock_exp)
                comet_cb.on_train_begin(args, state, control)
                comet_cb.on_log(args, state, control, logs={"eval_loss": 0.5})
                comet_cb.on_train_end(args, state, control)
                with Then("comet experiment logs metrics and closes"):
                    assert mock_exp.log_metric.called
                    assert mock_exp.end.called


def test_integration_registration_and_factory():
    with Scenario("Unit: Custom Integration Registration and Callback Resolution"):
        with Given("a custom IntegrationCallback subclass"):
            class DummyIntegration(IntegrationCallback):
                @classmethod
                def is_available(cls):
                    return True

                def on_log(self, args, state, control, logs=None, **kwargs):
                    pass

            with When("registering custom integration"):
                register_integration("dummy_integration", DummyIntegration)
                with Then("it can be resolved via get_reporting_integration_callbacks"):
                    cbs = get_reporting_integration_callbacks(["dummy_integration"])
                    assert len(cbs) == 1
                    assert isinstance(cbs[0], DummyIntegration)

            with When("resolving with report_to='all' and report_to='none'"):
                cbs_all = get_reporting_integration_callbacks("all")
                cbs_none = get_reporting_integration_callbacks("none")
                with Then("all and none behave according to specification"):
                    assert isinstance(cbs_all, list)
                    assert cbs_none == []
