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

from kilat.models.transformer import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.engine.args import TrainingArguments
from kilat.engine.callbacks import CallbackHandler, TrainerState
from kilat.engine.checkpoint import (
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


def test_checkpoint_sha256_manifest_and_bitwise_resume_parity():
    import json
    from kilat.data.collator import KilatDataCollator
    from kilat.data.dataloader import build_train_dataloader
    from kilat.data.dataset import SyntheticTokenDataset
    from kilat.engine.trainer import KilatTrainer

    with Scenario("Unit: SHA-256 Checkpoint Manifest & Bitwise Exact Resume Parity"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ds = SyntheticTokenDataset(vocab_size=48, seq_len=12, num_samples=24, seed=123)
            collator = KilatDataCollator(pad_token_id=0)

            cfg = KilatConfig(
                vocab_size=48,
                n_embd=32,
                n_layer=2,
                n_head=4,
                embd_drop=0.1,
                resid_drop=0.1,
                attn_drop=0.1,
            )

            with Given("Trajectory A trained uninterrupted for 6 steps"):
                dir_a = tmp_path / "run_a"
                dl_a = build_train_dataloader(ds, batch_size=4, collate_fn=collator, shuffle=True, seed=77)
                args_a = TrainingArguments(
                    output_dir=str(dir_a),
                    training_mode="steps",
                    max_steps=6,
                    learning_rate=1e-3,
                    logging_steps=1,
                    save_checkpoints=False,
                    precision="fp32",
                    seed=77,
                )
                KilatTrainer._set_seed(77)
                model_a = KilatTransformer(cfg)
                trainer_a = KilatTrainer(model_a, args_a, dl_a)
                state_a = trainer_a.train()
                losses_a = [entry["loss"] for entry in state_a.log_history if "loss" in entry]

            with When("Trajectory B saves at step 3 (same 6-step schedule, preempted at step 3) and resumes 4..6"):
                from kilat.engine.callbacks import TrainerCallback

                class _StopAtStep3(TrainerCallback):
                    def on_step_end(self, args, state, control, **kwargs):
                        if state.global_step >= 3:
                            control.should_training_stop = True

                dir_b = tmp_path / "run_b"
                dl_b1 = build_train_dataloader(ds, batch_size=4, collate_fn=collator, shuffle=True, seed=77)
                args_b1 = TrainingArguments(
                    output_dir=str(dir_b),
                    training_mode="steps",
                    max_steps=6,
                    learning_rate=1e-3,
                    logging_steps=1,
                    save_steps=3,
                    save_checkpoints=True,
                    precision="fp32",
                    seed=77,
                )
                KilatTrainer._set_seed(77)
                model_b1 = KilatTransformer(cfg)
                trainer_b1 = KilatTrainer(model_b1, args_b1, dl_b1, callbacks=[_StopAtStep3()])
                trainer_b1.train()

                ckpt3 = dir_b / "checkpoint-3"
                with open(ckpt3 / "training_args.json") as f:
                    saved_args_json = json.load(f)
                assert saved_args_json["learning_rate"] == 1e-3
                assert (ckpt3 / "checkpoint_manifest.json").exists()

                # Fresh model, dataloader, and trainer resuming from step 3 -> 6
                dl_b2 = build_train_dataloader(ds, batch_size=4, collate_fn=collator, shuffle=True, seed=999)
                args_b2 = TrainingArguments(
                    output_dir=str(dir_b),
                    resume_from_checkpoint=str(ckpt3),
                    training_mode="steps",
                    max_steps=6,
                    learning_rate=1e-3,
                    logging_steps=1,
                    save_checkpoints=False,
                    precision="fp32",
                    seed=999,
                )
                KilatTrainer._set_seed(999)
                model_b2 = KilatTransformer(cfg)
                trainer_b2 = KilatTrainer(model_b2, args_b2, dl_b2)
                state_b2 = trainer_b2.train()
                losses_b = [entry["loss"] for entry in state_b2.log_history if "loss" in entry]

                with Then("resumed losses and final parameters match uninterrupted training with 0.0 bitwise diff"):
                    assert losses_b == losses_a
                    for (name_a, p_a), (name_b, p_b) in zip(
                        model_a.named_parameters(), model_b2.named_parameters()
                    ):
                        assert name_a == name_b
                        max_diff = float((p_a - p_b).abs().max().item())
                        assert max_diff == 0.0, f"Bitwise mismatch on {name_a}: {max_diff}"

            with When("corrupting 1 byte in model.safetensors inside checkpoint-3"):
                weights_file = ckpt3 / "model.safetensors"
                if not weights_file.exists():
                    weights_file = ckpt3 / "pytorch_model.bin"
                raw_bytes = bytearray(weights_file.read_bytes())
                raw_bytes[-1] ^= 0xFF
                weights_file.write_bytes(bytes(raw_bytes))

                with Then("load_checkpoint immediately raises ERR_CHECKPOINT_CHECKSUM_MISMATCH"):
                    with pytest.raises(RuntimeError, match="ERR_CHECKPOINT_CHECKSUM_MISMATCH"):
                        load_checkpoint(
                            model=model_b2,
                            optimizer=trainer_b2.optimizer,
                            scheduler=trainer_b2.scheduler,
                            scaler=None,
                            state=TrainerState(),
                            callback_handler=CallbackHandler([]),
                            checkpoint_path=str(ckpt3),
                            device=torch.device("cpu"),
                        )
