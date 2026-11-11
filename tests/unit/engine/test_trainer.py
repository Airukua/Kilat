"""
Unit tests for KilatTrainer training loop and checkpoint management.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kilat.models.transformer import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.engine.args import TrainingArguments
from kilat.engine.trainer import KilatTrainer
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_kilat_trainer_training_loop():
    with Scenario("Unit: KilatTrainer Execution Loop"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "trainer_checkpoints"

            with Given("a small KilatTransformer model, dataset, and TrainingArguments"):
                torch.manual_seed(42)
                config = KilatConfig(
                    vocab_size=128,
                    n_embd=64,
                    n_layer=2,
                    n_head=4,
                    recall_ratio=0.5,
                    ffn_mode="dense",
                )
                model = KilatTransformer(config)

                args = TrainingArguments(
                    output_dir=str(out_dir),
                    training_mode="steps",
                    max_steps=2,
                    learning_rate=1e-3,
                    logging_steps=1,
                    save_steps=2,
                    precision="fp32",
                    report_to="none",
                )

                input_ids = torch.randint(0, 128, (4, 8))
                dataset = TensorDataset(input_ids)

                def dummy_collate(batch):
                    x = torch.stack([item[0] for item in batch])
                    return {"input_ids": x, "labels": x}

                dataloader = DataLoader(dataset, batch_size=2, collate_fn=dummy_collate)

                with When("KilatTrainer is initialized and train() is executed"):
                    trainer = KilatTrainer(
                        model=model,
                        args=args,
                        train_dataloader=dataloader,
                    )
                    trainer.train()

                    with Then("the trainer completes max_steps=2"):
                        assert trainer.state.global_step == 2

                    with And("checkpoint artifacts are safely saved to the output directory"):
                        assert out_dir.exists()
                        checkpoint_dirs = list(out_dir.glob("checkpoint-*"))
                        assert len(checkpoint_dirs) >= 1


def test_kilat_trainer_epoch_mode_and_evaluation():
    with Scenario("Unit: KilatTrainer Epoch Mode and Evaluation Loop"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "epochs_run"

            with Given("a configured KilatTransformer with evaluation dataset"):
                config = KilatConfig(
                    vocab_size=32,
                    n_embd=16,
                    n_layer=1,
                    n_head=2,
                    max_seq_len=16,
                )
                model = KilatTransformer(config)

                args = TrainingArguments(
                    output_dir=str(out_dir),
                    training_mode="epochs",
                    num_train_epochs=1,
                    learning_rate=1e-3,
                    logging_steps=1,
                    eval_steps=1,
                    precision="fp32",
                    report_to="none",
                    save_checkpoints=False,
                )

                def collate(b):
                    x = torch.stack([it[0] for it in b])
                    return {"input_ids": x, "labels": x}

                ds = TensorDataset(torch.randint(0, 32, (4, 8)))
                train_dl = DataLoader(ds, batch_size=2, collate_fn=collate)
                eval_dl = DataLoader(ds, batch_size=2, collate_fn=collate)

                with When("running epoch-based training"):
                    trainer = KilatTrainer(
                        model=model,
                        args=args,
                        train_dataloader=train_dl,
                        eval_dataloader=eval_dl,
                    )
                    state = trainer.train()

                    with Then("training terminates after completing epoch"):
                        assert state.global_step > 0
                        assert state.epoch >= 1.0

                with When("running explicit evaluation pass"):
                    metrics = trainer._evaluate()

                    with Then("eval metrics include eval_loss and perplexity"):
                        assert "eval_loss" in metrics
                        assert "perplexity" in metrics

                with When("checking metric improvement logic"):
                    is_better = trainer._is_better_metric(metrics["eval_loss"] - 0.1)
                    with Then("lower loss is detected as an improvement"):
                        assert is_better is True


def test_m06_pure_moe_perplexity_partial_accum_and_empty_target_guard():
    with Scenario("Unit: M-06 Pure MoE Perplexity, Partial Accumulation Scaling, & Empty Target Guard"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with Given("an MoE KilatTransformer with large aux_loss_coef=0.5 and z_loss_coef=1e-4"):
                import math
                from kilat.engine.optimizer import compute_total_steps

                torch.manual_seed(11)
                cfg = KilatConfig(
                    vocab_size=64,
                    n_embd=32,
                    n_layer=4,
                    n_head=4,
                    ffn_mode="moe",
                    num_experts=4,
                    active_experts=2,
                    num_shared_experts=0,
                    aux_loss_coef=0.5,
                    z_loss_coef=1e-4,
                )
                model = KilatTransformer(cfg)

                with When("evaluating an all-(-100) masked batch and inspecting residual depth scaling"):
                    ids = torch.randint(0, 64, (2, 8))
                    masked_labels = torch.full((2, 8), -100, dtype=torch.long)
                    out_masked = model(input_ids=ids, labels=masked_labels)
                    c_proj_std = float(model.layers[0].attn.c_proj.weight.std().item())
                    expected_res_std = 0.02 / math.sqrt(2 * 4)

                    with Then("masked batch loss is finite (no 0/0 NaN) and residual projections are depth-scaled"):
                        assert out_masked.loss is not None
                        assert torch.isfinite(out_masked.loss)
                        assert abs(c_proj_std - expected_res_std) < 0.005

                with When("running evaluation and partial-window epoch training (5 batches, grad_acc=2)"):
                    # Batch 1 has 7*2=14 valid targets, Batch 2 has 2*2=4 valid targets (padded with -100)
                    b1_ids = torch.randint(0, 64, (2, 8))
                    b1_lbl = b1_ids.clone()
                    b2_ids = torch.randint(0, 64, (2, 8))
                    b2_lbl = b2_ids.clone()
                    b2_lbl[:, 3:] = -100
                    eval_batches = [
                        {"input_ids": b1_ids, "labels": b1_lbl},
                        {"input_ids": b2_ids, "labels": b2_lbl},
                    ]
                    train_batches = [
                        {"input_ids": torch.randint(0, 64, (2, 8)), "labels": torch.randint(0, 64, (2, 8))}
                        for _ in range(5)
                    ]

                    args = TrainingArguments(
                        output_dir=str(Path(tmp_dir) / "m06"),
                        training_mode="epochs",
                        num_train_epochs=1,
                        gradient_accumulation_steps=2,
                        learning_rate=1e-3,
                        precision="fp32",
                        report_to="none",
                        save_checkpoints=False,
                    )
                    trainer = KilatTrainer(
                        model=model,
                        args=args,
                        train_dataloader=train_batches,  # type: ignore[arg-type]
                        eval_dataloader=eval_batches,  # type: ignore[arg-type]
                    )
                    total_planned = compute_total_steps(
                        training_mode="epochs",
                        max_steps=-1,
                        num_train_epochs=1,
                        dataloader_len=5,
                        gradient_accumulation_steps=2,
                    )
                    state = trainer.train()
                    metrics = trainer._evaluate()

                    # Verify exact pure CE token-weighted NLL directly
                    model.eval()
                    with torch.no_grad():
                        _ = model(**eval_batches[0])
                        nll1 = float(model._last_loss_breakdown["nll_sum"].item())  # type: ignore[index]
                        tok1 = int(model._last_loss_breakdown["valid_tokens"].item())  # type: ignore[index]
                        _ = model(**eval_batches[1])
                        nll2 = float(model._last_loss_breakdown["nll_sum"].item())  # type: ignore[index]
                        tok2 = int(model._last_loss_breakdown["valid_tokens"].item())  # type: ignore[index]
                    expected_eval_ce = (nll1 + nll2) / float(tok1 + tok2)

                    with Then("total steps matches ceil(5/2)=3 and eval_ppl equals exp(pure_token_weighted_ce)"):
                        assert total_planned == 3
                        assert state.global_step == 3
                        assert abs(metrics["eval_loss"] - expected_eval_ce) < 1e-5
                        assert abs(metrics["perplexity"] - math.exp(expected_eval_ce)) < 1e-3
                        assert "eval_aux_loss" in metrics and metrics["eval_aux_loss"] > 0.1
