"""
kilat.integrations.huggingface.data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hugging Face Datasets to Kilat binary memory-map bridge.
Streams and tokenizes raw HF dataset records directly into flat .npy/.bin
arrays optimized for Kilat's PretrainingDataset and CausalMemmapDataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np


def convert_hf_dataset_to_memmap(
    dataset: Any,
    tokenizer: Any,
    output_path: Union[str, Path],
    text_column: str = "text",
    dtype: Optional[np.dtype] = None,
    batch_size: int = 1000,
    append_eos: bool = True,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream and tokenize a Hugging Face Dataset directly into a binary .npy file.

    Parameters
    ----------
    dataset : Any
        Hugging Face Dataset or IterableDataset instance.
    tokenizer : Any
        Tokenizer instance exposing encode() or __call__().
    output_path : Union[str, Path]
        Target .npy file path.
    text_column : str, default="text"
        Column name in dataset containing text strings.
    dtype : Optional[np.dtype], default=None
        Target NumPy dtype. If None, auto-selects uint16 if vocab <= 65535, else int32.
    batch_size : int, default=1000
        Number of samples to batch tokenize.
    append_eos : bool, default=True
        Whether to append the EOS token after each sample document.
    max_tokens : Optional[int], default=None
        Maximum total tokens to collect.

    Returns
    -------
    Dict[str, Any]
        Dictionary with metadata: output_path, total_tokens, dtype, num_samples.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine vocabulary size and appropriate dtype
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    if dtype is None:
        dtype = np.uint16 if vocab_size <= 65535 else np.int32

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None and append_eos:
        eos_id = 2  # standard default if unspecified

    token_buffer: list[int] = []
    total_tokens = 0
    num_samples = 0

    def process_texts(texts: list[str]) -> bool:
        nonlocal total_tokens, num_samples
        if not texts:
            return True

        if hasattr(tokenizer, "batch_encode_plus"):
            encoded = tokenizer.batch_encode_plus(texts, add_special_tokens=False)["input_ids"]
        else:
            encoded = [tokenizer.encode(t, add_special_tokens=False) for t in texts]

        for sample_tokens in encoded:
            if not sample_tokens:
                continue
            token_buffer.extend(sample_tokens)
            if append_eos and eos_id is not None:
                token_buffer.append(eos_id)
            num_samples += 1

            if max_tokens is not None and len(token_buffer) >= max_tokens:
                del token_buffer[max_tokens:]
                total_tokens = len(token_buffer)
                return False
        return True

    batch: list[str] = []
    for item in dataset:
        text = item[text_column] if isinstance(item, dict) else getattr(item, text_column, "")
        if text:
            batch.append(text)
            if len(batch) >= batch_size:
                should_continue = process_texts(batch)
                batch = []
                if not should_continue:
                    break

    if batch and (max_tokens is None or len(token_buffer) < max_tokens):
        process_texts(batch)

    total_tokens = len(token_buffer)
    token_array = np.array(token_buffer, dtype=dtype)
    np.save(str(out_file), token_array)

    return {
        "output_path": str(out_file),
        "total_tokens": total_tokens,
        "dtype": str(dtype),
        "num_samples": num_samples,
    }
