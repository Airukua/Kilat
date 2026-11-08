"""
Unit tests for KilatTrainer training loop and checkpoint management.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from kilat.arc.model import KilatTransformer
from kilat.configs.model_config import KilatConfig
from kilat.training.args import TrainingArguments
from kilat.training.trainer import KilatTrainer
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
