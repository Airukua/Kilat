"""
Unit tests for extended dataset components and utilities.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
import warnings
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kilat.data.dataset import (
    ConcatDataset,
    KilatDataset,
    PackedDataset,
    PretrainingDataset,
    StreamingDataset,
    _discover_memmap_files,
    _discover_parquet_files,
    _to_list,
)
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_dataset_discovery_and_helpers():
    with Scenario("Unit: Dataset Discovery & Helper Functions"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("parquet and memmap files in a directory"):
                (tmp_path / "part-1.parquet").touch()
                (tmp_path / "part-2.parquet").touch()
                (tmp_path / "data-1.npy").touch()

                with When("discovering parquet files"):
                    pq_files = _discover_parquet_files(str(tmp_path))
                    with Then("all parquet files are discovered and sorted"):
                        assert len(pq_files) == 2
                        assert "part-1.parquet" in pq_files[0]

                with When("discovering memmap files"):
                    mm_files = _discover_memmap_files(str(tmp_path))
                    with Then("npy files are found"):
                        assert len(mm_files) == 1
                        assert "data-1.npy" in mm_files[0]

        with Given("different token container types"):
            list_tokens = [10, 20, 30]
            np_tokens = np.array([40, 50, 60])
            torch_tokens = torch.tensor([70, 80, 90])

            with When("converting to Python lists using _to_list"):
                with Then("all inputs are converted to valid python integer lists"):
                    assert _to_list(list_tokens, 0) == [10, 20, 30]
                    assert _to_list(np_tokens, 0) == [40, 50, 60]
                    assert _to_list(torch_tokens, 0) == [70, 80, 90]


def test_packed_and_concat_datasets():
    with Scenario("Unit: PackedDataset and ConcatDataset Operations"):
        with Given("a list of variable length token sequences"):
            sequences = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]

            with When("packing sequences into fixed block size blocks"):
                packed = PackedDataset(
                    sequences,
                    block_size=5,
                    eos_token_id=2,
                    pad_token_id=0,
                )
                with Then("PackedDataset packs tokens and exposes labels and attention info"):
                    assert len(packed) > 0
                    item = packed[0]
                    assert "input_ids" in item
                    assert "labels" in item
                    assert len(item["input_ids"]) == 5

            with When("concatenating datasets"):
                concat = ConcatDataset([packed, packed])
                with Then("length is cumulative and indexing spans all datasets"):
                    assert len(concat) == len(packed) * 2
                    first = concat[0]
                    assert len(first["input_ids"]) == 5


def test_pretraining_and_kilat_dataset():
    with Scenario("Unit: PretrainingDataset with Memory and Deprecated KilatDataset"):
        with Given("in-memory sequences"):
            memory_data = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]

            with When("loading PretrainingDataset from in-memory list"):
                ds = PretrainingDataset(source=memory_data, chunk_size=2)
                with Then("dataset slices items properly"):
                    assert len(ds) == 2
                    assert ds[0]["input_ids"] == [1, 2, 3, 4]

            with When("instantiating deprecated KilatDataset"):
                with warnings.catch_warnings(record=True) as recorded:
                    warnings.simplefilter("always")
                    kd = KilatDataset(file_or_data=memory_data)
                    with Then("deprecation warning is triggered and data is accessible"):
                        assert len(kd) > 0
                        assert any(issubclass(w.category, DeprecationWarning) for w in recorded)


def test_streaming_dataset_and_parquet_directory():
    with Scenario("Unit: StreamingDataset and PretrainingDataset Parquet Backend"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir) / "pq_data"
            dir_path.mkdir()
            pq_file = dir_path / "part1.parquet"

            with Given("a Parquet dataset containing token sequences"):
                data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
                table = pa.Table.from_arrays([pa.array(data)], names=["input_ids"])
                pq.write_table(table, pq_file)

                with When("initializing PretrainingDataset pointing to parquet directory"):
                    ds_dir = PretrainingDataset(str(dir_path), chunk_size=3)
                    with Then("it loads the sequences from parquet files"):
                        assert len(ds_dir) == 2
                        assert ds_dir[0]["input_ids"] == [1, 2, 3, 4, 5, 6]

                with When("streaming sequences using StreamingDataset"):
                    sds = StreamingDataset(str(pq_file), chunk_size=2, shuffle=True, drop_last=False)
                    sds.set_epoch(1)
                    chunks = list(iter(sds))
                    with Then("it iteratively streams fixed chunk blocks"):
                        assert len(chunks) == 6  # 12 tokens // 2 = 6 chunks
                        assert len(chunks[0]["input_ids"]) == 2


def test_pretraining_dataset_json_and_memmap_backends():
    with Scenario("Unit: PretrainingDataset Memmap and In-Memory Backends"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a single .npy file memmap"):
                npy_file = tmp_path / "tokens.npy"
                np.save(str(npy_file), np.arange(50, dtype=np.int64))

                with When("loading PretrainingDataset with .npy"):
                    ds_npy = PretrainingDataset(str(npy_file), chunk_size=10)
                    with Then("memmap chunks are indexed directly"):
                        assert len(ds_npy) == 5
                        assert ds_npy[0]["input_ids"] == list(range(10))

            with Given("in-memory list of token lists"):
                raw_lists = [[1, 2, 3, 4], [5, 6, 7, 8]]
                with When("loading PretrainingDataset with list of lists"):
                    ds_list = PretrainingDataset(raw_lists, chunk_size=2)
                    with Then("samples are accessible"):
                        assert len(ds_list) == 2
                        assert ds_list[0]["input_ids"] == [1, 2, 3, 4]

            with Given("invalid path formats or types"):
                with When("source is empty list"):
                    with Then("ValueError is raised"):
                        with pytest.raises(ValueError):
                            PretrainingDataset([])

                with When("source is integer"):
                    with And("TypeError is raised"):
                        with pytest.raises(TypeError):
                            PretrainingDataset(12345)


def test_concat_dataset_weighted_and_packed_variations():
    with Scenario("Unit: ConcatDataset Weighted Sampling & PackedDataset Sources"):
        with Given("two in-memory PretrainingDatasets"):
            data1 = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
            data2 = [{"input_ids": [7, 8, 9]}, {"input_ids": [10, 11, 12]}]
            ds1 = PretrainingDataset(data1, chunk_size=3)
            ds2 = PretrainingDataset(data2, chunk_size=3)

            with When("creating weighted ConcatDataset"):
                weighted_concat = ConcatDataset([ds1, ds2], weights=[0.5, 0.5], total_size=4, seed=42)
                with Then("it yields exactly total_size items"):
                    assert len(weighted_concat) == 4
                    assert len(weighted_concat[0]["input_ids"]) == 3

            with When("handling ConcatDataset error edge cases"):
                with Then("empty datasets raises ValueError"):
                    with pytest.raises(ValueError):
                        ConcatDataset([])

                with And("weights length mismatch raises ValueError"):
                    with pytest.raises(ValueError):
                        ConcatDataset([ds1, ds2], weights=[1.0])

                with And("out of range index raises IndexError"):
                    with pytest.raises(IndexError):
                        weighted_concat[100]

        with Given("a sequence longer than block size"):
            long_seqs = [[1] * 20]
            with When("packing into small block_size"):
                packed_long = PackedDataset(long_seqs, block_size=8, eos_token_id=2, pad_token_id=0)
                with Then("long sequences are truncated to block_size"):
                    assert len(packed_long[0]["input_ids"]) == 8

        with Given("KilatDataset with streaming=True"):
            with When("initializing deprecated KilatDataset with streaming=True"):
                with Then("ValueError is raised"):
                    with pytest.raises(ValueError, match="streaming=True is no longer supported"):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            KilatDataset([[1, 2, 3]], streaming=True)
