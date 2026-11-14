"""
Tests for Hugging Face Dataset to Kilat Memmap ingestion bridge.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from kilat.integrations.huggingface.data import convert_hf_dataset_to_memmap
from kilat.data.dataset import PretrainingDataset


class DummyTokenizer:
    """Minimal mock tokenizer for testing."""

    def __init__(self) -> None:
        self.vocab_size = 1000
        self.eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Simple word length hash
        return [len(word) + 10 for word in text.split()]

    def batch_encode_plus(self, texts: list[str], add_special_tokens: bool = False) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(t) for t in texts]}


def test_convert_hf_dataset_to_memmap_and_load() -> None:
    # Simulated HF dataset (list of dicts)
    raw_data = [
        {"text": "Once upon a time there was a little bird."},
        {"text": "The bird liked to sing in the bright morning sun."},
        {"text": "One day a cat came along quietly."},
    ]
    tokenizer = DummyTokenizer()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "tinystories_test.npy"
        meta = convert_hf_dataset_to_memmap(
            dataset=raw_data,
            tokenizer=tokenizer,
            output_path=out_file,
            text_column="text",
            batch_size=2,
            append_eos=True,
        )

        assert out_file.exists()
        assert meta["num_samples"] == 3
        assert meta["total_tokens"] > 10

        # Load back via NumPy
        arr = np.load(out_file)
        assert len(arr) == meta["total_tokens"]

        # Load via Kilat PretrainingDataset
        ds = PretrainingDataset(
            source=str(out_file),
            chunk_size=8,
            key_name="tokens",
        )
        assert len(ds) > 0
        sample = ds[0]
        assert "input_ids" in sample or "tokens" in sample or isinstance(sample, dict)
