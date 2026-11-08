"""
Unit tests for Kilat Data Pipeline: Collator, Dataset, and DataLoader builder.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import PretrainingDataset
from kilat.data.dataloader import (
    build_dataloader,
    build_eval_dataloader,
    build_train_dataloader,
    get_rank,
    get_world_size,
    is_distributed,
    worker_init_fn,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_kilat_data_collator():
    with Scenario("Unit: KilatDataCollator Padding & Loss Masking"):
        with Given("a KilatDataCollator with pad_token_id=0 and ignore_index=-100"):
            collator = KilatDataCollator(pad_token_id=0, max_length=16, ignore_index=-100)

            with When("collating heterogeneous variable-length sequences"):
                raw_batch = [
                    {"input_ids": [10, 20, 30, 40]},
                    {"input_ids": [50, 60]},
                ]
                batch = collator(raw_batch)

                with Then("the batch tensors have shape [2, 4]"):
                    assert batch["input_ids"].shape == (2, 4)
                    assert batch["labels"].shape == (2, 4)

                with And("the shorter sequence has left padding token 0"):
                    # Check that padding token is present in the shorter sequence
                    assert (batch["input_ids"][1] == 0).any()

                with And("the labels at padding positions contain ignore_index (-100)"):
                    assert (batch["labels"][1] == -100).any()
                    # Real tokens in labels match input_ids
                    real_mask = batch["labels"][1] != -100
                    assert torch.equal(batch["labels"][1][real_mask], batch["input_ids"][1][real_mask])

            with When("passing invalid inputs to KilatDataCollator"):
                with Then("appropriate exceptions are raised"):
                    with pytest.raises(ValueError):
                        collator([])
                    with pytest.raises(KeyError):
                        collator([{"tokens": [1, 2]}])
                    with pytest.raises(RuntimeError):
                        collator([{"input_ids": []}])
                    with pytest.raises(ValueError):
                        KilatDataCollator(pad_token_id=-2)
                    with pytest.raises(TypeError):
                        collator([{"input_ids": 12345}])

            with When("passing tensors, numpy arrays and truncating sequences"):
                truncated_collator = KilatDataCollator(pad_token_id=0, max_length=4)
                mixed_batch = [
                    {"input_ids": torch.tensor([1, 2, 3])},
                    {"input_ids": np.array([1, 2, 3, 4, 5, 6])},  # Truncates to 4
                ]
                out_batch = truncated_collator(mixed_batch)
                with Then("batch is properly converted and truncated"):
                    assert out_batch["input_ids"].shape == (2, 4)


def test_pretraining_dataset_memmap():
    with Scenario("Unit: PretrainingDataset NumPy Memmap Slicing"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "tokens.npy"

            with Given("a deterministic token array stored as a NumPy binary file"):
                tokens = np.arange(100, dtype=np.int64)
                np.save(file_path, tokens)

                with When("PretrainingDataset is initialized with chunk_size=10"):
                    dataset = PretrainingDataset(source=str(file_path), chunk_size=10, dtype=np.int64)

                    with Then("the total dataset length is exactly 10 chunks"):
                        assert len(dataset) == 10

                    with And("accessing item 0 returns the first 10 consecutive tokens"):
                        item_0 = dataset[0]
                        assert "input_ids" in item_0
                        assert len(item_0["input_ids"]) == 10
                        assert np.array_equal(item_0["input_ids"][:5], np.array([0, 1, 2, 3, 4]))

                    with And("accessing item 1 returns the next 10 consecutive tokens"):
                        item_1 = dataset[1]
                        assert np.array_equal(item_1["input_ids"][:5], np.array([10, 11, 12, 13, 14]))


def test_build_dataloader_integration():
    with Scenario("Unit: DataLoader Factory Functionality"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "tokens.npy"
            tokens = np.arange(50, dtype=np.int64)
            np.save(file_path, tokens)

            with Given("a valid dataset and collator"):
                dataset = PretrainingDataset(source=str(file_path), chunk_size=10, dtype=np.int64)
                collator = KilatDataCollator(pad_token_id=0)

                with When("build_train_dataloader is constructed with batch_size=2"):
                    train_loader = build_train_dataloader(
                        dataset=dataset,
                        batch_size=2,
                        collate_fn=collator,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                    )

                    with Then("the DataLoader produces batches of size 2"):
                        first_batch = next(iter(train_loader))
                        assert first_batch["input_ids"].shape[0] == 2
                        assert first_batch["input_ids"].shape[1] == 10


def test_dataloader_distributed_helpers_and_eval_loader():
    with Scenario("Unit: Distributed Helpers and Evaluation DataLoader"):
        with Given("distributed helper queries outside DDP environment"):
            with When("querying is_distributed, get_world_size, and get_rank"):
                dist_flag = is_distributed()
                world_sz = get_world_size()
                rank_id = get_rank()
                with Then("non-distributed defaults are returned (1 process, rank 0)"):
                    assert dist_flag is False
                    assert world_sz == 1
                    assert rank_id == 0

            with When("calling worker_init_fn"):
                worker_init_fn(worker_id=1, base_seed=42)
                with Then("it completes without errors"):
                    assert True

        with Given("a small dummy dataset"):
            data = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
            collator = KilatDataCollator(pad_token_id=0)

            with When("building an eval dataloader"):
                eval_loader = build_eval_dataloader(
                    dataset=data,
                    batch_size=2,
                    collate_fn=collator,
                    pin_memory=torch.cuda.is_available(),
                )
                with Then("the eval loader yields expected batch"):
                    b = next(iter(eval_loader))
                    assert b["input_ids"].shape[0] == 2
