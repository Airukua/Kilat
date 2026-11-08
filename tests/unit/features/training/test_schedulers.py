"""
Unit tests for Kilat learning rate schedulers.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch
import torch.nn as nn

from kilat.training.scheduler import (
    CosineScheduler,
    LinearScheduler,
    LRScheduler,
    SchedulerType,
    get_scheduler,
    register_scheduler,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_standard_schedulers_via_factory():
    with Scenario("Unit: Standard LR Schedulers Factory and Progression"):
        with Given("a simple model and optimizer"):
            model = nn.Linear(10, 10)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            scheduler_types = [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "cosine_with_min_lr",
                "polynomial",
                "constant",
                "constant_with_warmup",
                "inverse_sqrt",
                "wsdlr",
                "rex",
            ]

            with When("creating and stepping through each scheduler type"):
                for sched_name in scheduler_types:
                    opt = torch.optim.SGD(model.parameters(), lr=1.0)
                    sched = get_scheduler(
                        name=sched_name,
                        optimizer=opt,
                        num_warmup_steps=5,
                        num_training_steps=20,
                    )
                    with Then(f"scheduler '{sched_name}' initializes and computes learning rates"):
                        assert sched is not None
                        opt.step()
                        sched.step()
                        last_lr = sched.get_last_lr()
                        assert len(last_lr) == 1


def test_scheduler_state_dict_and_custom_registration():
    with Scenario("Unit: Scheduler State Persistence & Custom Registration"):
        with Given("an initialized LinearScheduler"):
            model = nn.Linear(5, 5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            sched = LinearScheduler(optimizer, num_warmup_steps=2, num_training_steps=10)

            with When("stepping and capturing state_dict"):
                optimizer.step()
                sched.step()
                state = sched.state_dict()
                with Then("state contains step information"):
                    assert "last_epoch" in state or "current_step" in state or len(state) > 0

            with When("loading state dict into fresh scheduler"):
                new_sched = LinearScheduler(optimizer, num_warmup_steps=2, num_training_steps=10)
                new_sched.load_state_dict(state)
                with Then("the state is successfully restored"):
                    assert new_sched.get_last_lr() == sched.get_last_lr()

        with Given("a custom scheduler class"):
            class CustomStepLR(LRScheduler):
                def get_lr_lambda(self):
                    return lambda step: 0.5

            with When("registering custom scheduler"):
                register_scheduler("custom_step", CustomStepLR)
                retrieved = get_scheduler(
                    "custom_step",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=10,
                )
                with Then("it is instantiable via get_scheduler"):
                    assert isinstance(retrieved, CustomStepLR)

            with When("registering with invalid inputs"):
                with Then("built-in name raises ValueError"):
                    with pytest.raises(ValueError, match="built‑in"):
                        register_scheduler("cosine", CustomStepLR)

                with And("non-subclass raises TypeError"):
                    with pytest.raises(TypeError):
                        register_scheduler("not_subclass", int)

                with And("duplicate registration raises ValueError"):
                    with pytest.raises(ValueError, match="already registered"):
                        register_scheduler("custom_step", CustomStepLR)


def test_scheduler_validation_and_kwargs():
    with Scenario("Unit: Scheduler Parameter Validation and Kwargs"):
        with Given("an optimizer"):
            model = nn.Linear(4, 4)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)

            with When("passing invalid boundary values to scheduler"):
                with Then("warmup < 0 raises ValueError"):
                    with pytest.raises(ValueError):
                        LinearScheduler(opt, num_warmup_steps=-1, num_training_steps=10)

                with And("training steps < 1 raises ValueError"):
                    with pytest.raises(ValueError):
                        LinearScheduler(opt, num_warmup_steps=0, num_training_steps=0)

                with And("warmup > total raises ValueError"):
                    with pytest.raises(ValueError):
                        LinearScheduler(opt, num_warmup_steps=15, num_training_steps=10)

            with When("creating scheduler with extra kwargs"):
                p_sched = get_scheduler(
                    "polynomial",
                    optimizer=opt,
                    num_warmup_steps=2,
                    num_training_steps=10,
                    power=2.0,
                    lr_end_ratio=0.01,
                )
                with Then("polynomial scheduler initializes"):
                    assert p_sched is not None
