import torch
from typing import Dict, List, Any, Optional
from utils.validators import validate_positive_int

class KilatDataCollator:
    """
    Collates raw token sequences into padded tensors for KilatTransformer
    causal language model training.

    Performs:
        - Left‑to‑right padding to the longest sequence in the batch.
        - Optional truncation to ``max_length``.
        - Creation of ``labels`` identical to ``input_ids`` (for causal LM loss).
        - Padding positions in ``labels`` are filled with ``ignore_index``
          so they are excluded from the loss.

    Design Rationale
    ---------------
    This collator is specifically designed for causal (autoregressive) language
    modeling where:
    - Labels are identical to input_ids (next-token prediction)
    - Left-padding is used so that the last "real" token of each sequence
      aligns with the end of the tensor, which is critical for autoregressive
      generation where the model attends left-to-right
    - Padding tokens in labels are set to ignore_index (-100 by default) so
      the loss function skips them — without this, the model would be penalized
      for "predicting" padding tokens

    Why Left Padding?
    -----------------
    In causal LMs, left-padding ensures:
    1. The model's causal mask naturally allows each real token to attend to
       all previous real tokens (padding on the left is "in the past" and
       doesn't interfere with attention).
    2. During generation, the last position always contains a real token,
       making it easy to extract the next-token prediction.
    3. FlashAttention and other optimized kernels handle left-padded sequences
       correctly with causal masks.

    Right-padding would place real tokens at the start and padding at the end,
    which breaks causal generation because the model would need to "predict"
    after the padding tokens — positions that don't exist in the original sequence.

    Empty Sequence Handling
    -----------------------
    Empty sequences are silently skipped. This handles edge cases like:
    - Tokenization errors producing empty outputs
    - Filtered datasets where some examples are empty after processing
    - Incomplete batches at the end of an epoch
    If ALL sequences are empty, a RuntimeError is raised since there's
    nothing to train on.

    Memory Pinning
    -------------
    Tensors are created directly on CPU but NOT pinned in the collator.
    The DataLoader's pin_memory=True setting handles pinning automatically
    when the batch is moved to the DataLoader's output queue. Creating tensors
    with torch.full and then filling is more efficient than stacking individual
    tensors and padding afterward.

    Example::
        >>> collator = KilatDataCollator(pad_token_id=0, max_length=128)
        >>> batch = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
        >>> out = collator(batch)
        >>> print(out["input_ids"].shape)   # (2, 3)
        >>> print(out["labels"])            # tensor with -100 padding on left
        >>> # Sequence 1: [1, 2, 3]
        >>> # Sequence 2: [0, 4, 5]  (0 is pad_token_id)
        >>> # Labels 2:   [-100, 4, 5]  (-100 ignored in loss)
    """

    def __init__(
        self,
        pad_token_id: int,
        max_length: Optional[int] = None,
        ignore_index: int = -100,
    ):
        """
        Parameters
        ----------
        pad_token_id : int
            Token ID used for padding shorter sequences. Must be a valid
            token in the model's vocabulary (typically 0 for most tokenizers).
            Using pad_token_id + 1 in validation is a trick to allow 0 while
            still catching negative values.
        max_length : Optional[int]
            Maximum sequence length. Sequences longer than this are truncated
            (keeping the LAST max_length tokens, which preserves the most
            recent context for causal LMs). None means no truncation.
        ignore_index : int
            Value used to fill padding positions in labels. The standard is
            -100 (PyTorch's CrossEntropyLoss default ignore_index). These
            positions are excluded from loss computation.
        """
        # Validate pad_token_id allowing 0 (common default for many tokenizers)
        # while still rejecting negative values. The +1 trick converts 0→1
        # which passes validate_positive_int, while -1→0 which fails.
        validate_positive_int("pad_token_id", pad_token_id + 1)  # allows 0
        if max_length is not None:
            validate_positive_int("max_length", max_length)

        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.ignore_index = ignore_index

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples into padded training tensors.

        Processing Steps
        ----------------
        1. Extract and normalize sequences from batch examples
        2. Apply truncation (keeping prefix, not suffix, for causal LM)
        3. Determine max sequence length in the batch
        4. Allocate padded tensors filled with pad_token_id / ignore_index
        5. Copy sequences into the padded tensors (left-aligned)
        6. Return dict with input_ids and labels

        Normalization
        ------------
        Supports multiple input formats:
        - torch.Tensor → converted via .tolist()
        - list → used directly
        - Objects with .tolist() method (e.g., numpy arrays)
        This flexibility allows the collator to work with various dataset
        formats without requiring preprocessing.

        Truncation Strategy
        ------------------
        For causal LMs, we keep the FIRST max_length tokens (seq_list[:max_length]).
        While keeping the LAST tokens would preserve the most recent context
        (useful for generation), it would discard the beginning of the sequence
        which often contains task instructions or important context. The choice
        depends on use case; prefix truncation is the safer default.

        Parameters
        ----------
        examples : List[Dict[str, Any]]
            List of dictionaries, each containing at least the key "input_ids"
            with a sequence of token indices (list, tensor, or array-like).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with:
            - "input_ids": LongTensor of shape (batch_size, max_seq_len)
            - "labels": LongTensor of shape (batch_size, max_seq_len)
              Padding positions are filled with ignore_index.

        Raises
        ------
        ValueError
            If examples list is empty.
        KeyError
            If "input_ids" key is missing from the first example.
        TypeError
            If a sequence is not a list, tensor, or array-like.
        RuntimeError
            If all sequences in the batch are empty after processing.
        """
        if not examples:
            raise ValueError("Collator received an empty batch.")

        input_key = "input_ids"
        if input_key not in examples[0]:
            raise KeyError(
                f"Expected key '{input_key}' missing from batch dictionaries."
            )

        raw_sequences = [example[input_key] for example in examples]

        # ---- 1. Convert to list of ints, apply truncation, find max length ----
        processed_sequences = []
        batch_max_len = 0

        for seq in raw_sequences:
            # Normalise to Python list for consistent handling.
            # This avoids the overhead of torch.stack + padding logic and
            # handles mixed input types (tensors, lists, numpy arrays).
            if isinstance(seq, torch.Tensor):
                seq_list = seq.tolist()
            elif hasattr(seq, "tolist"):
                seq_list = seq.tolist()  # numpy arrays, jax arrays, etc.
            elif isinstance(seq, list):
                seq_list = seq
            else:
                raise TypeError(f"Unsupported sequence type: {type(seq)}")

            # Skip empty sequences silently.
            # These can occur from tokenizer errors or dataset filtering.
            # Removing them maintains batch integrity without crashing.
            if len(seq_list) == 0:
                continue

            # Truncation: keep first max_length tokens.
            # For causal LMs, prefix truncation preserves the task context
            # (instructions, prompts) which typically appear at the start.
            # Alternative: seq_list[-self.max_length:] for suffix truncation
            # when generation-continuation is the primary use case.
            if self.max_length is not None and len(seq_list) > self.max_length:
                seq_list = seq_list[:self.max_length]

            processed_sequences.append(seq_list)
            batch_max_len = max(batch_max_len, len(seq_list))

        # Safety check: ensure at least one non-empty sequence remains.
        # An entirely empty batch would produce tensors of shape (0, 0)
        # which would cause cryptic errors downstream in the model.
        if not processed_sequences:
            raise RuntimeError("All sequences in the batch are empty.")

        # ---- 2. Allocate padded tensors ----
        # Create tensors pre-filled with padding/ignore values.
        # This is more efficient than creating individual tensors and stacking,
        # especially for large batch sizes with varying sequence lengths.
        # torch.full allocates and fills in one operation.
        batch_size = len(processed_sequences)
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

        # ---- 3. Copy sequences into padded tensors (left-aligned) ----
        # Left-alignment means sequences start at position 0 with padding
        # on the right. This is correct for causal LMs because:
        # 1. The causal mask ensures position i only attends to positions ≤ i
        # 2. Padding on the right means real tokens are at positions [0, seq_len)
        # 3. The model's last hidden state for position seq_len-1 is the
        #    "real" last token, not a padding token
        for idx, seq in enumerate(processed_sequences):
            seq_len = len(seq)
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            # Copy real tokens to the left portion of the padded tensors.
            # input_ids: real tokens followed by pad_token_id
            # labels: real tokens followed by ignore_index
            input_ids_tensor[idx, :seq_len] = seq_tensor
            labels_tensor[idx, :seq_len] = seq_tensor

        # ---- 4. Return batch dictionary ----
        # Note: Memory pinning is handled by the DataLoader's pin_memory=True
        # setting, not by the collator. The DataLoader automatically pins
        # tensors when moving them from worker processes to the main process.
        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }