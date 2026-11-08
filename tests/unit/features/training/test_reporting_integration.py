"""
Unit tests for Kilat Reporting Integrations and ConsoleCallback.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest

from kilat.engine.integrations import (
    get_reporting_integration_callbacks,
    ConsoleCallback,
    IntegrationCallback,
)
from kilat.engine.args import TrainingArguments
from kilat.engine.callbacks import TrainerState, TrainerControl
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_reporting_callback_factory():
    with Scenario("Unit: Reporting Integration Callbacks Factory"):
        with Given("different report_to configuration strings"):
            with When("report_to is 'none'"):
                cbs = get_reporting_integration_callbacks("none")
                with Then("an empty callback list is returned"):
                    assert cbs == []

            with When("report_to is None"):
                cbs_none = get_reporting_integration_callbacks(None)
                with Then("an empty list is returned"):
                    assert cbs_none == []

            with When("report_to is a list of backend strings"):
                cbs_list = get_reporting_integration_callbacks(["none"])
                with Then("it resolves to an empty list"):
                    assert cbs_list == []


def test_console_callback_execution():
    with Scenario("Unit: Console Callback Event Dispatch"):
        with Given("a ConsoleCallback instance and training state"):
            cb = ConsoleCallback()
            args = TrainingArguments(output_dir="./test_out", precision="fp32")
            state = TrainerState(global_step=5, epoch=1)
            control = TrainerControl()

            with When("on_log is dispatched with loss metrics"):
                cb.on_log(args, state, control, logs={"loss": 3.14, "lr": 1e-4})
                with Then("the callback processes the log event without error"):
                    assert True

            with When("on_evaluate is dispatched with evaluation metrics"):
                cb.on_evaluate(args, state, control, metrics={"eval_loss": 2.71})
                with Then("the callback processes the evaluation event"):
                    assert True
