"""
Unit tests for Kilat Parquet to Memmap Converter Pipeline.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kilat.data.preprocess import (
    _discover_parquet_files,
    _count_tokens_parquet,
    parquet_to_memmap,
    parquet_to_memmap_sharded,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_parquet_to_memmap_pipeline():
    with Scenario("Unit: Parquet to Memmap Conversion"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            parquet_path = tmp_path / "sample.parquet"
            memmap_out = tmp_path / "output.npy"

            with Given("a valid Apache Parquet file with token ID lists"):
                token_data = [[10, 20, 30], [40, 50], [60, 70, 80, 90]]
                table = pa.Table.from_arrays(
                    [pa.array(token_data, type=pa.list_(pa.int64()))],
                    names=["input_ids"],
                )
                pq.write_table(table, str(parquet_path))

                with When("discovering parquet files from directory"):
                    discovered = _discover_parquet_files(str(parquet_path))
                    with Then("the file is properly discovered in a list"):
                        assert len(discovered) == 1
                        assert discovered[0] == str(parquet_path)

                with When("counting total tokens from parquet"):
                    token_count = _count_tokens_parquet(discovered, "input_ids", 100, False)
                    with Then("the token count accurately reflects all 9 tokens"):
                        assert token_count == 9

                with When("parquet_to_memmap is executed"):
                    parquet_to_memmap(
                        input_path=str(parquet_path),
                        output_path=str(memmap_out),
                        key_name="input_ids",
                        dtype=np.int64,
                        force_overwrite=True,
                        verbose=False,
                    )

                    with Then("the .npy memmap file is generated"):
                        assert memmap_out.exists()

                    with And("loading the memmap array yields all 9 tokens in order"):
                        loaded_tokens = np.load(str(memmap_out))
                        assert len(loaded_tokens) == 9
                        assert np.array_equal(loaded_tokens, np.array([10, 20, 30, 40, 50, 60, 70, 80, 90]))
