"""
Unit tests for Parquet to memmap conversion pipeline.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kilat.pipeline.converter.to_memmap import (
    _discover_parquet_files,
    parquet_to_memmap,
    parquet_to_memmap_sharded,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_parquet_to_memmap_conversion():
    with Scenario("Unit: Parquet to Memmap Conversion"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a sample Parquet file with token sequences"):
                pq_file = tmp_path / "sample.parquet"
                out_memmap = tmp_path / "tokens.npy"

                data = [[10, 20, 30], [40, 50], [60, 70, 80, 90]]
                table = pa.Table.from_arrays([pa.array(data)], names=["input_ids"])
                pq.write_table(table, pq_file)

                with When("discovering files in directory"):
                    discovered = _discover_parquet_files(str(tmp_path))
                    with Then("the sample parquet file is listed"):
                        assert len(discovered) == 1

                with When("converting Parquet file to memmap"):
                    parquet_to_memmap(
                        input_path=str(pq_file),
                        output_path=str(out_memmap),
                        key_name="input_ids",
                        verbose=False,
                    )

                    with Then("the output memmap file is generated"):
                        assert out_memmap.exists()

                    with And("tokens in memmap match source data"):
                        loaded_arr = np.load(out_memmap, mmap_mode="r")
                        assert len(loaded_arr) == 3 + 2 + 4
                        assert list(loaded_arr[:3]) == [10, 20, 30]


def test_parquet_to_memmap_sharded_not_implemented():
    with Scenario("Unit: Parquet to Memmap Sharding Exception"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pq_file = tmp_path / "data.parquet"
            out_dir = tmp_path / "shards"

            table = pa.Table.from_arrays([pa.array([[1, 2]])], names=["input_ids"])
            pq.write_table(table, pq_file)

            with Given("an input parquet file"):
                with When("invoking parquet_to_memmap_sharded"):
                    with Then("NotImplementedError is raised advising parquet_to_memmap"):
                        with pytest.raises(NotImplementedError):
                            parquet_to_memmap_sharded(str(pq_file), str(out_dir), verbose=False)
