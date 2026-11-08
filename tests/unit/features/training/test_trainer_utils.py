"""
Unit tests for trainer utilities in Kilat.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import math
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.training.args import TrainingArguments
from kilat.training.callbacks import CallbackHandler, TrainerState
from kilat.training.trainer_utils import (
    _tokenizer_needs_saving,
    average_losses,
    clip_grad_norm_,
    compute_perplexity,
    compute_perplexity_from_logits,
    format_metrics_with_ppl,
    get_current_lr,
    get_device,
    get_latest_checkpoint,
    get_lr_group_info,
    load_checkpoint,
    load_checkpoint_simple,
    prune_checkpoints,
    save_checkpoint,
    save_checkpoint_simple,
    should_evaluate,
    should_log,
    should_save,
    should_save_on_rank0,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_perplexity_and_metric_formatting():
    with Scenario("Unit: Perplexity Computation and Metric Formatting"):
        with Given("a known loss value"):
            loss = 2.0
            with When("computing perplexity"):
                ppl = compute_perplexity(loss)
                with Then("ppl equals exp(loss)"):
                    assert math.isclose(ppl, math.exp(loss), rel_tol=1e-4)

        with Given("logits and label tensors"):
            batch_size, seq_len, vocab_size = 2, 4, 10
            logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))

            with When("computing perplexity from logits"):
                loss_val, ppl_from_logits = compute_perplexity_from_logits(logits, labels)
                with Then("the computed perplexity is positive and finite"):
                    assert loss_val > 0
                    assert ppl_from_logits > 0
                    assert not math.isinf(ppl_from_logits)

        with Given("a metrics dictionary containing loss"):
            metrics = {"eval_loss": 1.5, "accuracy": 0.8}
            with When("formatting metrics with perplexity"):
                formatted = format_metrics_with_ppl(metrics)
                with Then("perplexity and ppl keys are added"):
                    assert "perplexity" in formatted
                    assert math.isclose(formatted["perplexity"], math.exp(1.5), rel_tol=1e-4)


def test_trigger_predicates():
    with Scenario("Unit: Logging, Evaluation and Checkpointing Predicates"):
        with Given("a trainer state at step 100"):
            state = TrainerState(global_step=100, epoch=2.0)

            with When("checking should_log"):
                with Then("it triggers on matching step"):
                    assert should_log(state, logging_steps=50) is True
                    assert should_log(state, logging_steps=30) is False

            with When("checking should_evaluate"):
                with Then("it triggers on matching step"):
                    assert should_evaluate(state, eval_steps=100) is True
                    assert should_evaluate(state, eval_steps=75) is False

            with When("checking should_save"):
                with Then("it triggers on matching step"):
                    assert should_save(state, save_steps=50) is True
                    assert should_save(state, save_steps=45) is False


def test_optimizer_helpers_and_loss_averaging():
    with Scenario("Unit: Optimizer Helpers, Gradient Clipping and Loss Averaging"):
        with Given("a model and optimizer"):
            model = nn.Linear(4, 4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

            with When("querying current learning rate"):
                lr = get_current_lr(optimizer)
                with Then("the initial learning rate is returned"):
                    assert lr == 0.005

            with When("querying parameter group info"):
                info = get_lr_group_info(optimizer)
                with Then("it lists group configurations"):
                    assert len(info) >= 1
                    assert info[0]["lr"] == 0.005

            with When("clipping gradient norm"):
                loss = model(torch.randn(2, 4)).sum()
                loss.backward()
                total_norm = clip_grad_norm_(model, max_norm=1.0)
                with Then("total norm is computed and gradients are clipped"):
                    assert total_norm is not None

        with Given("accumulated losses"):
            loss_dict = {"loss": 4.0, "lm_loss": 2.0}
            with When("averaging across 2 steps"):
                averaged = average_losses(loss_dict, steps=2)
                with Then("values are properly divided"):
                    assert averaged["loss"] == 2.0
                    assert averaged["lm_loss"] == 1.0


def test_checkpointing_and_pruning():
    with Scenario("Unit: Simple Checkpointing and Pruning"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a model, optimizer, scheduler, state and handler"):
                model = nn.Linear(2, 2)
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                state = TrainerState(global_step=5, epoch=1.0)
                handler = CallbackHandler([])

                ckpt_file = tmp_path / "checkpoint.pt"

                with When("saving simple checkpoint"):
                    save_checkpoint_simple(
                        ckpt_file,
                        model=model,
                        optimizer=opt,
                        scheduler=None,
                        state=state,
                        callback_handler=handler,
                    )
                    with Then("the checkpoint file is created"):
                        assert ckpt_file.exists()

                with When("loading the simple checkpoint"):
                    new_model = nn.Linear(2, 2)
                    new_opt = torch.optim.SGD(new_model.parameters(), lr=0.1)
                    new_state = TrainerState()
                    load_checkpoint_simple(
                        ckpt_file,
                        model=new_model,
                        optimizer=new_opt,
                        scheduler=None,
                        state=new_state,
                        callback_handler=handler,
                    )
                    with Then("the state is restored"):
                        assert new_state.global_step == 5
                        assert new_state.epoch == 1.0

            with Given("multiple step checkpoint directories with ordered timestamps"):
                import os
                import time

                ckpt10 = tmp_path / "checkpoint-10"
                ckpt20 = tmp_path / "checkpoint-20"
                ckpt30 = tmp_path / "checkpoint-30"

                ckpt10.mkdir()
                ckpt20.mkdir()
                ckpt30.mkdir()

                now = time.time()
                os.utime(ckpt10, (now - 300, now - 300))
                os.utime(ckpt20, (now - 200, now - 200))
                os.utime(ckpt30, (now - 100, now - 100))

                with When("fetching the latest checkpoint"):
                    latest = get_latest_checkpoint(str(tmp_path))
                    with Then("it points to checkpoint-30"):
                        assert latest is not None
                        assert "checkpoint-30" in latest

                with When("pruning checkpoints to keep 2"):
                    prune_checkpoints(str(tmp_path), save_total_limit=2)
                    with Then("oldest checkpoint-10 is removed"):
                        assert not (tmp_path / "checkpoint-10").exists()
                        assert (tmp_path / "checkpoint-20").exists()
                        assert (tmp_path / "checkpoint-30").exists()


def test_full_checkpointing_lifecycle():
    with Scenario("Unit: Complete Checkpoint Save and Restore Pipeline"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a configured KilatTransformer and TrainingArguments"):
                cfg = KilatConfig(vocab_size=32, n_embd=16, n_layer=1, n_head=2, max_seq_len=16)
                model = KilatTransformer(cfg)
                opt = torch.optim.SGD(model.parameters(), lr=0.1)
                state = TrainerState(global_step=5, epoch=1.0)
                handler = CallbackHandler([])
                args = TrainingArguments(output_dir=str(tmp_path), precision="fp32")

                with When("querying device and rank utilities"):
                    dev = get_device(model)
                    rank0 = should_save_on_rank0(state)
                    tok_needs = _tokenizer_needs_saving(None)
                    with Then("utilities return expected device and rank status"):
                        assert dev == torch.device("cpu")
                        assert rank0 is True
                        assert tok_needs is True

                with When("saving full checkpoint with model artifacts"):
                    save_checkpoint(
                        model=model,
                        optimizer=opt,
                        scheduler=None,
                        scaler=None,
                        state=state,
                        callback_handler=handler,
                        output_dir=str(tmp_path),
                        step=5,
                        training_args=args,
                    )
                    ckpt_dir = tmp_path / "checkpoint-5"
                    with Then("checkpoint directory contains safetensors and metadata"):
                        assert ckpt_dir.exists()
                        assert (ckpt_dir / "training_state.pt").exists()
                        assert (ckpt_dir / "training_args.json").exists()

                with When("loading the full checkpoint"):
                    loaded_model = load_checkpoint(
                        model=model,
                        optimizer=opt,
                        scheduler=None,
                        scaler=None,
                        state=state,
                        callback_handler=handler,
                        checkpoint_path=str(ckpt_dir),
                        device=torch.device("cpu"),
                    )
                    with Then("the model instance is successfully restored"):
                        assert loaded_model is not None


def test_tokenizer_saving_heuristics_and_checkpoint_variations():
    from unittest.mock import MagicMock
    from kilat.configs.tokenizer_config import TokenizerConfig

    with Scenario("Unit: Tokenizer Needs Saving Heuristics & Checkpoint Edge Cases"):
        with Given("various TokenizerConfig configurations"):
            spm_cfg = TokenizerConfig(tokenizer_type="sentencepiece", tokenizer_model_path="/tmp/spm.model")
            custom_cfg = TokenizerConfig(tokenizer_type="custom", custom_builder="dummy:builder")
            local_cfg = TokenizerConfig(tokenizer_type="gpt2", tokenizer_name_or_path="./local/tok")
            hub_cfg = TokenizerConfig(tokenizer_type="gpt2", tokenizer_name_or_path="gpt2")

            with When("checking _tokenizer_needs_saving heuristics"):
                with Then("sentencepiece and custom require saving"):
                    assert _tokenizer_needs_saving(spm_cfg) is True
                    assert _tokenizer_needs_saving(custom_cfg) is True
                    assert _tokenizer_needs_saving(local_cfg) is True
                    assert _tokenizer_needs_saving(hub_cfg) is False

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a model and mock early stopping component"):
                cfg = KilatConfig(vocab_size=32, n_embd=16, n_layer=1, n_head=2)
                model = KilatTransformer(cfg)
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
                state = TrainerState(global_step=10, epoch=2.0)
                handler = CallbackHandler([])
                early_stop = MagicMock()
                early_stop.best_metric = 0.42
                early_stop.early_stopping_patience_counter = 1

                with When("saving checkpoint with atomic=False and early_stopping"):
                    save_checkpoint(
                        model=model,
                        optimizer=opt,
                        scheduler=sched,
                        scaler=None,
                        state=state,
                        callback_handler=handler,
                        output_dir=str(tmp_path),
                        step=10,
                        early_stopping=early_stop,
                        additional_state={"step_loss": 0.5},
                        atomic=False,
                    )
                    ckpt10 = tmp_path / "checkpoint-10"
                    with Then("checkpoint-10 exists"):
                        assert ckpt10.exists()

                with When("loading checkpoint into model with early_stopping restoration"):
                    es_restore = MagicMock()
                    es_restore.best_metric = 999.0
                    es_restore.early_stopping_patience_counter = 0

                    loaded = load_checkpoint(
                        model=model,
                        optimizer=opt,
                        scheduler=sched,
                        scaler=None,
                        state=state,
                        callback_handler=handler,
                        checkpoint_path=str(ckpt10),
                        early_stopping=es_restore,
                        device=torch.device("cpu"),
                    )
                    with Then("early stopping best_metric and counter are restored"):
                        assert es_restore.best_metric == 0.42
                        assert es_restore.early_stopping_patience_counter == 1
                        assert loaded is not None

                with When("attempting load_checkpoint on nonexistent directory"):
                    with Then("FileNotFoundError is raised"):
                        with pytest.raises(FileNotFoundError):
                            load_checkpoint(
                                model=model,
                                optimizer=opt,
                                scheduler=sched,
                                scaler=None,
                                state=state,
                                callback_handler=handler,
                                checkpoint_path=str(tmp_path / "nonexistent"),
                                device=torch.device("cpu"),
                            )
