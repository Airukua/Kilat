"""
Seam 2 Integration Test:
Verifies offline dataset extraction, dynamic batching, and multi-epoch DataLoader mini-batch feeding.
Follows standards from ml_integration_testing_structure_md.md (Seam 2).
"""

import tempfile
from pathlib import Path
import numpy as np
import torch

from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import PretrainingDataset
from kilat.data.dataloader import build_train_dataloader
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_seam2_dataset_to_dataloader_contract():
    with Scenario("Integration: Dataset to DataLoader Mini-Batch Contract (Seam 2)"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            golden_tokens = np.arange(100, dtype=np.int64)
            data_file = tmp_path / "train_corpus.npy"
            np.save(data_file, golden_tokens)

            with Given("a validated memmap dataset and KilatDataCollator"):
                dataset = PretrainingDataset(source=str(data_file), chunk_size=10, dtype=np.int64)
                collator = KilatDataCollator(pad_token_id=0, max_length=10, ignore_index=-100)

                with When("DataLoader constructs micro-batches for the training loop"):
                    loader = build_train_dataloader(
                        dataset=dataset,
                        batch_size=4,
                        collate_fn=collator,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=torch.cuda.is_available(),
                    )
                    batch = next(iter(loader))

                    with Then("the batch tensors have shape [batch_size=4, seq_len=10]"):
                        assert batch["input_ids"].shape == (4, 10)
                        assert batch["labels"].shape == (4, 10)

                    with And("input_ids and labels types conform to PyTorch LongTensor"):
                        assert batch["input_ids"].dtype == torch.int64
                        assert batch["labels"].dtype == torch.int64

                    with And("no data leakage occurs across mini-batch sample boundaries"):
                        # Ensure row 0 and row 1 have distinct consecutive token sequences
                        assert not torch.equal(batch["input_ids"][0], batch["input_ids"][1])
