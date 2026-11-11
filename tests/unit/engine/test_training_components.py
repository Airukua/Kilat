"""
Unit tests for Kilat Training Components: TrainingArguments, Scheduler, Optimizer, Callbacks, and Utilities.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
import pytest
import torch
import torch.nn as nn

from kilat.engine.args import TrainingArguments
from kilat.engine.optimizer import (
    compute_total_steps,
    create_optimizer,
    resolve_amp_dtype,
)
from kilat.engine.scheduler import get_scheduler
from kilat.engine.callbacks import TrainerCallback, CallbackHandler, EarlyStoppingCallback, TrainerState, TrainerControl
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_training_arguments_validation():
    with Scenario("Unit: TrainingArguments Validation & Defaults"):
        with Given("valid argument parameters for steps mode"):
            args = TrainingArguments(
                output_dir="./test_checkpoints",
                training_mode="steps",
                max_steps=100,
                learning_rate=2e-4,
                precision="fp32",
            )
            with When("checking initialized properties"):
                with Then("learning_rate is 2e-4"):
                    assert args.learning_rate == 2e-4
                with And("training_mode is steps"):
                    assert args.training_mode == "steps"
                with And("max_steps is 100"):
                    assert args.max_steps == 100

        with Given("invalid arguments parameters"):
            with When("precision is fp16 on CPU without CUDA"):
                if not torch.cuda.is_available():
                    with Then("a ValueError is raised"):
                        with pytest.raises(ValueError):
                            TrainingArguments(output_dir="./test", precision="fp16")

            with When("testing invalid training_mode or steps/epochs bounds"):
                with Then("ValueErrors are raised"):
                    with pytest.raises(ValueError):
                        TrainingArguments(training_mode="bad_mode")
                    with pytest.raises(ValueError):
                        TrainingArguments(training_mode="steps", max_steps=0)
                    with pytest.raises(ValueError):
                        TrainingArguments(training_mode="epochs", num_train_epochs=0)
                    with pytest.raises(ValueError):
                        TrainingArguments(precision="unknown")


def test_optimizer_creation_and_decay_grouping():
    with Scenario("Unit: Parameter Grouping & Optimizer Creation"):
        with Given("a toy neural network with weights and biases"):
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.LayerNorm(32),
                nn.Linear(32, 2),
            )
            with When("create_optimizer is invoked with weight_decay=0.01"):
                optimizer = create_optimizer(
                    model=model,
                    learning_rate=1e-3,
                    weight_decay=0.01,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8,
                )
                with Then("an AdamW optimizer is returned with two parameter groups"):
                    assert len(optimizer.param_groups) == 2
                with And("the first group has weight_decay=0.01"):
                    assert optimizer.param_groups[0]["weight_decay"] == 0.01
                with And("the second group (biases & LayerNorm) has weight_decay=0.0"):
                    assert optimizer.param_groups[1]["weight_decay"] == 0.0

        with Given("a full KilatTransformer with Global Decay log_lambda and MLA LayerNorms"):
            from kilat.models.transformer import KilatTransformer
            from kilat.configs.model_config import KilatConfig

            cfg = KilatConfig(
                vocab_size=128,
                n_embd=32,
                n_layer=2,
                n_head=4,
                recall_ratio=0.5,
                use_rope=True,
                rope_head_dim=8,
            )
            kilat_model = KilatTransformer(cfg)
            opt = create_optimizer(kilat_model, learning_rate=3e-4, weight_decay=0.1)
            decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
            nodecay_ids = {id(p) for p in opt.param_groups[1]["params"]}

            with When("auditing parameter group assignment for 1D decay and norm parameters"):
                with Then("log_lambda, q_a_norm, kv_a_norm, rms1, rms2, and ln_f all receive weight_decay=0.0"):
                    for pname, param in kilat_model.named_parameters():
                        if any(k in pname for k in ("log_lambda", "q_a_norm", "kv_a_norm", "rms1", "rms2", "ln_f")):
                            assert id(param) in nodecay_ids, f"{pname} leaked into weight_decay group"
                            assert id(param) not in decay_ids


def test_scheduler_warmup_and_cosine_decay():
    with Scenario("Unit: Learning Rate Scheduler Trajectory"):
        with Given("an optimizer initialized at lr=1e-3"):
            param = nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([param], lr=1e-3)

            with When("get_scheduler creates a cosine schedule with 10 warmup steps out of 100 total"):
                scheduler = get_scheduler(
                    name="cosine",
                    optimizer=optimizer,
                    num_warmup_steps=10,
                    num_training_steps=100,
                )
                with Then("at step 0, learning rate starts at or near 0"):
                    initial_lr = scheduler.get_last_lr()[0]
                    assert initial_lr <= 1e-4

                with When("stepping through warmup to step 10"):
                    for _ in range(10):
                        optimizer.step()
                        scheduler.step()
                    with Then("learning rate reaches peak lr 1e-3"):
                        peak_lr = scheduler.get_last_lr()[0]
                        assert abs(peak_lr - 1e-3) < 1e-5

                with When("stepping through cosine decay to step 100"):
                    for _ in range(90):
                        optimizer.step()
                        scheduler.step()
                    with Then("learning rate decays towards minimum"):
                        final_lr = scheduler.get_last_lr()[0]
                        assert final_lr < 1e-4


def test_callbacks_and_early_stopping():
    with Scenario("Unit: Trainer Callback Handlers & Early Stopping"):
        with Given("a TrainerState, TrainerControl, and CallbackHandler"):
            args = TrainingArguments(output_dir="./test_out", precision="fp32")
            state = TrainerState()
            state.global_step = 10
            state.epoch = 1
            control = TrainerControl()

            early_stopper = EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )
            handler = CallbackHandler([early_stopper], args=args)

            with When("loss improves on first evaluation"):
                state.best_metric = 2.5
                control = handler.on_evaluate(state, control, metrics={"eval_loss": 2.5})
                with Then("training is not stopped"):
                    assert not control.should_training_stop

            with When("loss fails to improve for 2 consecutive evaluations"):
                control = handler.on_evaluate(state, control, metrics={"eval_loss": 2.6})
                control = handler.on_evaluate(state, control, metrics={"eval_loss": 2.7})
                with Then("early stopping triggers and should_training_stop becomes True"):
                    assert control.should_training_stop


def test_callback_handler_lifecycle_and_state():
    with Scenario("Unit: CallbackHandler Events Dispatch and Serialization"):
        with Given("a custom stateful callback registered in CallbackHandler"):
            class StatefulCallback(TrainerCallback):
                def __init__(self):
                    self.called_events = []
                    self.custom_metric = 99

                def on_train_begin(self, args, state, control, **kwargs):
                    self.called_events.append("train_begin")

                def on_step_end(self, args, state, control, **kwargs):
                    self.called_events.append("step_end")

                def state_dict(self):
                    return {"custom_metric": self.custom_metric}

                def load_state_dict(self, s):
                    self.custom_metric = s.get("custom_metric", 0)

            cb = StatefulCallback()
            args = TrainingArguments(output_dir="./test_tmp", precision="fp32")
            handler = CallbackHandler([cb], args=args)
            state = TrainerState(global_step=1, epoch=0.5)
            control = TrainerControl()

            with When("dispatching lifecycle hooks"):
                handler.on_train_begin(state, control)
                handler.on_epoch_begin(state, control)
                handler.on_step_begin(state, control)
                handler.on_step_end(state, control)
                handler.on_epoch_end(state, control)
                handler.on_train_end(state, control)

                with Then("the callback records the events"):
                    assert "train_begin" in cb.called_events
                    assert "step_end" in cb.called_events

            with When("serializing and reloading callback handler state"):
                saved_state = handler.state_dict()
                new_cb = StatefulCallback()
                new_handler = CallbackHandler([new_cb], args=args)
                new_handler.load_state_dict(saved_state)

                with Then("state is restored into new callback"):
                    assert new_cb.custom_metric == 99

            with When("managing callbacks via add, pop, and remove"):
                handler.add_callback(StatefulCallback())
                popped = handler.pop_callback(StatefulCallback)
                with Then("callback can be popped by class"):
                    assert isinstance(popped, StatefulCallback)


def test_optimizer_utils_and_amp_dtype():
    with Scenario("Unit: Optimizer Utilities, Step Computation, and AMP Resolution"):
        with Given("precision options"):
            with When("resolving amp dtypes"):
                with Then("fp16 maps to torch.float16"):
                    assert resolve_amp_dtype("fp16") == torch.float16
                with And("bf16 maps to torch.bfloat16"):
                    assert resolve_amp_dtype("bf16") == torch.bfloat16
                with And("fp32 maps to None"):
                    assert resolve_amp_dtype("fp32") is None

        with Given("training hyperparameters for step calculation"):
            with When("computing steps in 'steps' mode"):
                total_steps_mode = compute_total_steps(
                    training_mode="steps",
                    max_steps=500,
                    num_train_epochs=3,
                    dataloader_len=100,
                    gradient_accumulation_steps=2,
                )
                with Then("it returns max_steps directly"):
                    assert total_steps_mode == 500

            with When("computing steps in 'epochs' mode"):
                total_epochs_mode = compute_total_steps(
                    training_mode="epochs",
                    max_steps=-1,
                    num_train_epochs=4,
                    dataloader_len=100,
                    gradient_accumulation_steps=2,
                )
                with Then("it multiplies epochs by batches per accumulation window"):
                    # 100 // 2 = 50 steps per epoch * 4 epochs = 200 steps
                    assert total_epochs_mode == 200


def test_callback_handler_management_and_usage_example():
    from kilat.engine.callbacks import (
        _usage_example,
        TrainerCallback,
        CallbackHandler,
        EarlyStoppingCallback,
        TrainerControl,
        TrainerState,
    )

    with Scenario("Unit: CallbackHandler Management, Lifecycle Events & Usage Example"):
        with Given("a CallbackHandler instance"):
            cb1 = EarlyStoppingCallback(early_stopping_patience=2)
            handler = CallbackHandler(callbacks=[cb1])

            with When("testing remove_callback and pop_callback"):
                with Then("pop_callback returns the callback instance"):
                    popped = handler.pop_callback(EarlyStoppingCallback)
                    assert popped is cb1

                with And("popping nonexistent callback type raises ValueError"):
                    with pytest.raises(ValueError):
                        handler.pop_callback(EarlyStoppingCallback)

                handler.add_callback(cb1)
                handler.remove_callback(EarlyStoppingCallback)
                with And("removing nonexistent callback type raises ValueError"):
                    with pytest.raises(ValueError):
                        handler.remove_callback(EarlyStoppingCallback)

            with When("adding an invalid callback object"):
                with Then("TypeError is raised"):
                    with pytest.raises(TypeError):
                        handler.add_callback("not_a_callback")

            with When("testing EarlyStoppingCallback state_dict and load_state_dict"):
                es = EarlyStoppingCallback(early_stopping_patience=3, metric_for_best_model="loss")
                es.best_metric = 0.25
                es.early_stopping_patience_counter = 2
                sd = es.state_dict()
                with Then("state dict captures best_metric and counter"):
                    assert sd["best_metric"] == 0.25
                    assert sd["early_stopping_patience_counter"] == 2

                es2 = EarlyStoppingCallback(early_stopping_patience=3)
                es2.load_state_dict(sd)
                with And("load_state_dict restores values"):
                    assert es2.best_metric == 0.25
                    assert es2.early_stopping_patience_counter == 2

            with When("executing _usage_example helper"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _usage_example()
                    with Then("_usage_example runs without error"):
                        assert True
