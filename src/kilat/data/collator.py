from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
import numpy as np
import torch
from kilat.utils.validators import validate_choice, validate_positive_int


class KilatDataCollator:
    """
    Collates raw token sequences into padded ``torch.long`` tensors for causal LM training
    and batched generation, with a zero-copy fast path for uniform tensor/memmap batches.

    Padding Behaviour
    -----------------
    - ``padding_side="right"`` (default for pre-training): real tokens are placed at
      ``[0 : seq_len]`` and trailing positions are filled with ``pad_token_id`` in
      ``input_ids`` and ``ignore_index`` (``-100``) in ``labels``.
    - ``padding_side="left"`` (for batched autoregressive generation): padding is placed
      at ``[0 : batch_max_len - seq_len]`` so the final token of every sequence aligns
      at index ``-1``.
    """

    def __init__(
        self,
        pad_token_id: int,
        max_length: Optional[int] = None,
        ignore_index: int = -100,
        padding_side: Literal["right", "left"] = "right",
    ) -> None:
        validate_positive_int("pad_token_id", pad_token_id + 1)  # allows 0, rejects < 0
        if max_length is not None:
            validate_positive_int("max_length", max_length)
        validate_choice("padding_side", padding_side, ("right", "left"))

        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.padding_side: Literal["right", "left"] = padding_side

    def _to_1d_long_tensor(self, seq: Any) -> torch.Tensor:
        """Convert a single sequence to a 1D ``torch.long`` tensor without Python list round-trips."""
        if isinstance(seq, torch.Tensor):
            if seq.ndim != 1:
                raise TypeError(f"Expected 1D tensor sequence, got shape {tuple(seq.shape)}")
            return seq if seq.dtype == torch.long else seq.to(dtype=torch.long)

        if isinstance(seq, np.ndarray):
            if seq.ndim != 1:
                raise TypeError(f"Expected 1D numpy sequence, got shape {seq.shape}")
            return torch.as_tensor(seq, dtype=torch.long)

        if isinstance(seq, list):
            return torch.tensor(seq, dtype=torch.long)

        if hasattr(seq, "tolist"):
            converted = seq.tolist()
            if not isinstance(converted, list):
                raise TypeError(f"Unsupported scalar/non-list sequence type: {type(seq)}")
            return torch.tensor(converted, dtype=torch.long)

        raise TypeError(f"Unsupported sequence type: {type(seq)}")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("Collator received an empty batch.")

        input_key = "input_ids"
        if input_key not in examples[0]:
            raise KeyError(f"Expected key '{input_key}' missing from batch dictionaries.")

        tensors: List[torch.Tensor] = []
        label_tensors: List[Optional[torch.Tensor]] = []
        has_explicit_labels = "labels" in examples[0]
        batch_max_len = 0
        all_uniform = True
        first_len: Optional[int] = None

        for example in examples:
            t = self._to_1d_long_tensor(example[input_key])
            if t.numel() == 0:
                continue

            lbl: Optional[torch.Tensor] = None
            if has_explicit_labels and "labels" in example and example["labels"] is not None:
                lbl = self._to_1d_long_tensor(example["labels"])

            if self.max_length is not None and t.numel() > self.max_length:
                t = t[: self.max_length]
                if lbl is not None:
                    lbl = lbl[: self.max_length]

            seq_len = int(t.numel())
            if first_len is None:
                first_len = seq_len
            elif seq_len != first_len:
                all_uniform = False

            tensors.append(t)
            label_tensors.append(lbl)
            if seq_len > batch_max_len:
                batch_max_len = seq_len

        if not tensors:
            raise RuntimeError("All sequences in the batch are empty.")

        # Zero-copy / direct-stack fast path for uniform-length batches (e.g. memmap / synthetic / packed)
        if all_uniform:
            input_ids_tensor = torch.stack(tensors, dim=0)
            if has_explicit_labels and all(lbl is not None for lbl in label_tensors):
                labels_tensor = torch.stack([lbl for lbl in label_tensors if lbl is not None], dim=0)
            else:
                labels_tensor = input_ids_tensor.clone()
            return {
                "input_ids": input_ids_tensor,
                "labels": labels_tensor,
            }

        batch_size = len(tensors)
        input_ids_tensor = torch.full(
            (batch_size, batch_max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        labels_tensor = torch.full(
            (batch_size, batch_max_len),
            fill_value=self.ignore_index,
            dtype=torch.long,
        )

        for idx, t in enumerate(tensors):
            seq_len = int(t.numel())
            lbl = label_tensors[idx]
            target_seq = lbl if (lbl is not None and lbl.numel() == seq_len) else t
            if self.padding_side == "right":
                input_ids_tensor[idx, :seq_len] = t
                labels_tensor[idx, :seq_len] = target_seq
            else:
                input_ids_tensor[idx, -seq_len:] = t
                labels_tensor[idx, -seq_len:] = target_seq

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }
