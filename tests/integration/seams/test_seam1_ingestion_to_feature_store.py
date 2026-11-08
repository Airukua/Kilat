"""
Seam 1 Integration Test:
Verifies raw token ingestion and conversion into zero-copy memory-mapped shards (Memmap / Parquet).
Follows standards from ml_integration_testing_structure_md.md (Seam 1).
"""

import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kilat.data.dataset import PretrainingDataset
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_seam1_ingestion_to_memmap():
    with Scenario("Integration: Ingestion & Memmap Dataset Assembly (Seam 1)"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            golden_raw_tokens = np.array([101, 2054, 2003, 1037, 3231, 102], dtype=np.int64)
            shard_path = tmp_path / "shard_00000.npy"

            with Given("a continuous stream of raw token IDs from data preprocessing"):
                np.save(shard_path, golden_raw_tokens)

                with When("ingested into the PretrainingDataset memmap storage engine"):
                    dataset = PretrainingDataset(source=str(shard_path), chunk_size=3, dtype=np.int64)

                    with Then("the dataset maps exactly into deterministic chunks"):
                        assert len(dataset) == 2

                    with And("token sequences are byte-for-byte identical with raw input"):
                        chunk_0 = dataset[0]["input_ids"]
                        chunk_1 = dataset[1]["input_ids"]
                        reconstructed = np.concatenate([chunk_0, chunk_1])
                        assert np.array_equal(reconstructed, golden_raw_tokens)

                    with And("token types and memory layouts match ML training specifications"):
                        assert isinstance(chunk_0, list)
                        assert len(chunk_0) == 3
                        assert isinstance(chunk_0[0], int)
