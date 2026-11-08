from __future__ import annotations

import json
import os
import time
import zlib
from typing import Any, Iterable, Optional

import numpy as np
import torch

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult


def _extract_13gram_hashes(tokens_1d: list[int], n: int = 13) -> set[int]:
    """Compute fast deterministic CRC32/Adler-free 64-bit hashes over contiguous n-grams."""
    if not tokens_1d:
        return set()
    if len(tokens_1d) < n:
        # Only hash full short sequences if at least 8 tokens long to avoid trivial 2-token collisions
        if len(tokens_1d) < 8:
            return set()
        raw = np.asarray(tokens_1d, dtype=np.int64).tobytes()
        return {zlib.crc32(raw) & 0xFFFFFFFF}

    hashes: set[int] = set()
    arr = np.asarray(tokens_1d, dtype=np.int64)
    for start in range(0, len(tokens_1d) - n + 1):
        window_bytes = arr[start : start + n].tobytes()
        # Combine two seeded CRC32s into a 64-bit fingerprint to prevent birthday collisions
        h1 = zlib.crc32(window_bytes, 0) & 0xFFFFFFFF
        h2 = zlib.crc32(window_bytes, 0x9E3779B9) & 0xFFFFFFFF
        hashes.add((h1 << 32) | h2)
    return hashes


def _audit_split_batches(
    dataloader: Iterable[Any],
    vocab_size: int,
    max_batches: int,
    split_name: str,
) -> tuple[int, int, set[int]]:
    """Audit up to `max_batches` from `dataloader` and return `(batches_checked, valid_tokens_total, ngram_hashes)`."""
    batches_checked = 0
    valid_tokens_total = 0
    ngram_hashes: set[int] = set()

    for batch in dataloader:
        if batches_checked >= max_batches:
            break
        if not isinstance(batch, dict) or "input_ids" not in batch:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MASK_ALIGNMENT_BREACH,
                f"{split_name} batch #{batches_checked} is not a dict containing 'input_ids'.",
            )

        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        attention_mask = batch.get("attention_mask")

        if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise AssuranceError(
                AssuranceFailureCode.ERR_MASK_ALIGNMENT_BREACH,
                f"{split_name} batch #{batches_checked} 'input_ids'/'labels' must be torch.Tensor.",
            )

        if input_ids.dtype != torch.long or labels.dtype != torch.long:
            raise AssuranceError(
                AssuranceFailureCode.ERR_TOKEN_OUT_OF_BOUNDS,
                f"{split_name} batch #{batches_checked} expected torch.long dtype, "
                f"got input_ids={input_ids.dtype}, labels={labels.dtype}.",
            )

        if input_ids.ndim != 2 or input_ids.shape != labels.shape:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MASK_ALIGNMENT_BREACH,
                f"{split_name} batch #{batches_checked} shape mismatch: "
                f"input_ids={tuple(input_ids.shape)} vs labels={tuple(labels.shape)}.",
            )

        min_tok = int(input_ids.min().item()) if input_ids.numel() > 0 else 0
        max_tok = int(input_ids.max().item()) if input_ids.numel() > 0 else -1
        if input_ids.numel() == 0 or min_tok < 0 or max_tok >= vocab_size:
            raise AssuranceError(
                AssuranceFailureCode.ERR_TOKEN_OUT_OF_BOUNDS,
                f"{split_name} batch #{batches_checked} input_ids outside [0, {vocab_size - 1}] "
                f"(min={min_tok}, max={max_tok}).",
                {"split": split_name, "min_token": min_tok, "max_token": max_tok, "vocab_size": vocab_size},
            )

        valid_label_mask = labels != -100
        if bool(valid_label_mask.any().item()):
            valid_label_vals = labels[valid_label_mask]
            lbl_min = int(valid_label_vals.min().item())
            lbl_max = int(valid_label_vals.max().item())
            if lbl_min < 0 or lbl_max >= vocab_size:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_TOKEN_OUT_OF_BOUNDS,
                    f"{split_name} batch #{batches_checked} labels outside [0, {vocab_size - 1}] U {{-100}} "
                    f"(min={lbl_min}, max={lbl_max}).",
                )

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.shape != input_ids.shape:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MASK_ALIGNMENT_BREACH,
                    f"{split_name} batch #{batches_checked} attention_mask shape {tuple(attention_mask.shape)} "
                    f"!= input_ids shape {tuple(input_ids.shape)}.",
                )
            pad_positions = attention_mask == 0
            if bool(pad_positions.any().item()) and bool((labels[pad_positions] != -100).any().item()):
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MASK_ALIGNMENT_BREACH,
                    f"{split_name} batch #{batches_checked} has unmasked labels (!= -100) where attention_mask == 0.",
                )

        # Causal shift target count: labels[:, 1:] != -100
        shifted_valid = int((labels[:, 1:] != -100).sum().item()) if labels.shape[1] > 1 else 0
        if shifted_valid <= 0:
            raise AssuranceError(
                AssuranceFailureCode.ERR_EMPTY_TARGET_BATCH,
                f"{split_name} batch #{batches_checked} contains 0 valid causal target tokens (all labels[:, 1:] == -100).",
                {"split": split_name, "batch_index": batches_checked},
            )

        valid_tokens_total += shifted_valid

        # Extract 13-gram hashes on unpadded tokens per row
        for row_idx in range(input_ids.shape[0]):
            if isinstance(attention_mask, torch.Tensor):
                active_toks = input_ids[row_idx][attention_mask[row_idx] != 0].tolist()
            else:
                active_toks = input_ids[row_idx].tolist()
            ngram_hashes.update(_extract_13gram_hashes(active_toks, n=13))

        batches_checked += 1

    if batches_checked == 0:
        raise AssuranceError(
            AssuranceFailureCode.ERR_EMPTY_TARGET_BATCH,
            f"{split_name} dataloader yielded 0 batches.",
        )

    return batches_checked, valid_tokens_total, ngram_hashes


def _verify_memmap_file(memmap_path: str, dtype_name: str = "uint16") -> None:
    if not os.path.isfile(memmap_path):
        raise AssuranceError(
            AssuranceFailureCode.ERR_MEMMAP_INTEGRITY_BREACH,
            f"Memmap file does not exist: {memmap_path}",
        )
    np_dtype = np.dtype(dtype_name)
    byte_size = os.path.getsize(memmap_path)
    if memmap_path.endswith(".npy"):
        try:
            arr = np.load(memmap_path, mmap_mode="r")
        except Exception as exc:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MEMMAP_INTEGRITY_BREACH,
                f"Corrupted .npy memmap file '{memmap_path}': {exc}",
            ) from exc
        if arr.size == 0:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MEMMAP_INTEGRITY_BREACH,
                f"Memmap .npy file '{memmap_path}' contains 0 tokens.",
            )
        actual_tokens = int(arr.size)
    else:
        if byte_size == 0 or byte_size % np_dtype.itemsize != 0:
            raise AssuranceError(
                AssuranceFailureCode.ERR_MEMMAP_INTEGRITY_BREACH,
                f"Memmap file '{memmap_path}' byte size {byte_size} is empty or not divisible by itemsize {np_dtype.itemsize}.",
            )
        actual_tokens = byte_size // np_dtype.itemsize
    meta_candidates = [
        f"{memmap_path}.meta.json",
        os.path.splitext(memmap_path)[0] + ".meta.json",
        f"{os.path.splitext(memmap_path)[0]}.npy_meta.json",
    ]
    for meta_path in meta_candidates:
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            expected_tokens = meta.get("num_tokens", meta.get("total_tokens"))
            if expected_tokens is not None and int(expected_tokens) != actual_tokens:
                raise AssuranceError(
                    AssuranceFailureCode.ERR_MEMMAP_INTEGRITY_BREACH,
                    f"Memmap token count mismatch against '{meta_path}': expected {expected_tokens}, got {actual_tokens}.",
                )


def run_gate4_data(
    train_dataloader: Iterable[Any],
    vocab_size: int,
    eval_dataloader: Optional[Iterable[Any]] = None,
    max_audit_batches: int = 8,
    allow_train_val_overlap: bool = False,
    memmap_path: Optional[str] = None,
    memmap_dtype: str = "uint16",
) -> GateResult:
    """
    Gate 4: Data Pipeline & Leakage Forensics (`G4_DATA_FORENSICS`).
    Verifies:
      1. Token bounds [0, V-1], torch.long dtypes, mask alignment (`attention_mask==0 => labels==-100`),
         and non-empty shifted targets (`(labels[:, 1:] != -100).sum() > 0`).
      2. 13-gram rolling hash train-validation decontamination check.
      3. Optional `np.memmap` binary alignment & `.meta.json` consistency.
    """
    t0 = time.perf_counter()

    if memmap_path is not None:
        _verify_memmap_file(memmap_path, dtype_name=memmap_dtype)

    train_batches, train_valid_tokens, train_hashes = _audit_split_batches(
        train_dataloader,
        vocab_size=vocab_size,
        max_batches=max_audit_batches,
        split_name="train",
    )

    eval_batches = 0
    eval_valid_tokens = 0
    contaminated_ngrams = 0
    if eval_dataloader is not None:
        eval_batches, eval_valid_tokens, eval_hashes = _audit_split_batches(
            eval_dataloader,
            vocab_size=vocab_size,
            max_batches=max_audit_batches,
            split_name="eval",
        )
        overlap = train_hashes.intersection(eval_hashes)
        contaminated_ngrams = len(overlap)
        if contaminated_ngrams > 0 and not allow_train_val_overlap:
            raise AssuranceError(
                AssuranceFailureCode.ERR_TRAIN_VAL_CONTAMINATION,
                f"Detected {contaminated_ngrams} exact 13-gram sequence leaks between train and eval splits.",
                {"contaminated_13grams": contaminated_ngrams},
            )

    duration_ms = (time.perf_counter() - t0) * 1000.0
    return GateResult(
        gate_id="G4_DATA_FORENSICS",
        status="PASS",
        duration_ms=duration_ms,
        metrics={
            "train_batches_audited": train_batches,
            "train_valid_tokens": train_valid_tokens,
            "eval_batches_audited": eval_batches,
            "eval_valid_tokens": eval_valid_tokens,
            "contaminated_13grams": contaminated_ngrams,
        },
    )
