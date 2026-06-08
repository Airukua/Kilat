"""
Unified tokenizer wrapper for Kilat training pipelines.

Supports HuggingFace tokenizers (AutoTokenizer) and custom SentencePiece/BPE tokenizers
with a consistent interface for tokenization, padding, truncation, and serialization.

WHY THIS EXISTS:
    - KilatTrainer and datasets expect a tokenizer with a standard set of methods
      (tokenize, decode, pad_token_id, etc.).
    - Different projects may use different tokenizer implementations (HF transformers,
      custom SentencePiece, Tiktoken for GPT, etc.).
    - This wrapper provides a single abstraction layer, making the trainer agnostic
      to the underlying tokenizer library.
    - It also adds convenience methods for preprocessing text corpora into token
      sequences suitable for PretrainingDataset or StreamingDataset.

Design Decisions:
    - The class does NOT inherit from HuggingFace's PreTrainedTokenizerBase because
      we want to remain decoupled. Instead, it delegates to an internal tokenizer
      object that conforms to a minimal interface.
    - For HuggingFace tokenizers, we use AutoTokenizer.from_pretrained and wrap it.
    - For custom tokenizers (e.g., you trained your own SentencePiece model), we
      provide a simple wrapper that loads a SentencePieceProcessor.
    - Methods are designed to be robust to missing attributes (e.g., if the tokenizer
      has no pad_token_id, we infer from eos_token_id or set to 0).
    - All methods accept **kwargs to forward to the underlying tokenizer where
      appropriate (e.g., truncation, max_length, padding).

Performance Considerations:
    - Tokenization is usually the bottleneck in data preprocessing. This class does
      not add significant overhead; it delegates directly to the underlying tokenizer.
    - For large-scale preprocessing, use the `batch_tokenize` method which can be
      faster than tokenizing one by one if the underlying tokenizer supports batching.
    - The class caches attributes like pad_token_id, eos_token_id after first access.

Edge Cases:
    - If the underlying tokenizer does not have a `pad_token`, we set it to the
      `eos_token` (common for GPT-style models) and raise a warning.
    - If neither pad nor eos exists, we set pad_token_id = 0 and log a warning.
    - Tokenizing an empty string returns an empty list (not None or [pad_token]).
    - Truncation is applied before padding to avoid wasteful computation.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Lazy imports with availability flags – optional dependencies.
# WHY: Users should not be forced to install transformers or sentencepiece if they
# only need one of them. This pattern also allows the wrapper to work in environments
# where these packages are not present (e.g., inference-only setups), though the
# corresponding tokenizer types will not be available.
try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


class KilatTokenizer:
    """
    Unified tokenizer wrapper for Hugging Face and custom SentencePiece tokenizers.

    This class provides a common interface for tokenization, decoding, and
    configuration management. It also handles padding, truncation, and batch
    tokenization for preprocessing large corpora.

    Parameters
    ----------
    tokenizer : Any, optional
        An existing tokenizer object (e.g., from transformers or sentencepiece).
        If None, you must provide `pretrained_model_name_or_path`.
    pretrained_model_name_or_path : str, optional
        Hugging Face model name or local path to a tokenizer.
    tokenizer_type : str, optional
        One of 'hf' (HuggingFace) or 'sentencepiece'. If not provided, auto-detected
        from the tokenizer object or file extension.
    cache_dir : str, optional
        Directory to cache downloaded tokenizers.
    use_fast : bool, default True
        Whether to use the fast tokenizer (if available) for HuggingFace.
    additional_special_tokens : list, optional
        Additional special tokens to add to the tokenizer.
    **kwargs
        Extra arguments passed to the tokenizer initialisation.

    Attributes
    ----------
    pad_token_id : int
        Token ID used for padding.
    eos_token_id : int
        Token ID used for end-of-sequence.
    bos_token_id : int
        Token ID used for beginning-of-sequence (may be None).
    vocab_size : int
        Size of the tokenizer's vocabulary.

    Example Usage
    -------------
        >>> # HuggingFace tokenizer
        >>> tok = KilatTokenizer(pretrained_model_name_or_path="gpt2")
        >>> tokens = tok.tokenize("Hello world")
        >>> print(tokens)  # [15496, 995]
        >>>
        >>> # SentencePiece custom model
        >>> tok = KilatTokenizer(pretrained_model_name_or_path="./spm.model", tokenizer_type="sentencepiece")
        >>> tokens = tok.tokenize("Hello world")
    """

    def __init__(
        self,
        tokenizer: Any = None,
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_fast: bool = True,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        if tokenizer is None and pretrained_model_name_or_path is None:
            raise ValueError("Either `tokenizer` or `pretrained_model_name_or_path` must be provided.")

        self._tokenizer = None
        self._type = None
        self._cache_dir = cache_dir

        if tokenizer is not None:
            # User provided an already constructed tokenizer object.
            self._tokenizer = tokenizer
            self._type = self._detect_type(tokenizer)
        else:
            self._load_from_pretrained(
                pretrained_model_name_or_path,
                tokenizer_type,
                use_fast,
                **kwargs,
            )

        # Add additional special tokens if requested (e.g., for meta tokens like <|im_start|>)
        if additional_special_tokens:
            self.add_special_tokens(additional_special_tokens)

        # Cache token IDs; we will fill them in _infer_special_tokens
        self._pad_token_id = None
        self._eos_token_id = None
        self._bos_token_id = None
        self._vocab_size = None

        # Infer padding settings from underlying tokenizer. This also handles
        # the case where the tokenizer lacks a pad_token (common for GPT-style models).
        self._infer_special_tokens()

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _detect_type(self, tokenizer: Any) -> str:
        """
        Detect the type of the given tokenizer object.

        WHY: The user may pass a pre‑constructed tokenizer from various libraries.
        We need to know which API to use (HF or SentencePiece). This detection is
        heuristic but covers the two main cases.

        Heuristics:
        - HF tokenizers have `vocab_size` and `tokenize` method.
        - SentencePieceProcessor has `piece_to_id` and `EncodeAsIds`.
        - If none match, assume HF (most common) and emit a warning.

        Edge Cases: Custom tokenizer objects that mimic one of these APIs will work.
        If the user passes a completely unrelated object, the detection will likely
        produce wrong behaviour later – this is considered user error.
        """
        # HuggingFace tokenizers have a `vocab_size` attribute and `tokenize` method
        if hasattr(tokenizer, "vocab_size") and hasattr(tokenizer, "tokenize"):
            return "hf"
        # SentencePieceProcessor has `piece_to_id` and `EncodeAsIds`
        if hasattr(tokenizer, "piece_to_id") and hasattr(tokenizer, "EncodeAsIds"):
            return "sentencepiece"
        # Default to hf (most common)
        warnings.warn(f"Could not detect tokenizer type, assuming HuggingFace: {type(tokenizer)}")
        return "hf"

    def _load_from_pretrained(
        self,
        model_name: str,
        tokenizer_type: Optional[str],
        use_fast: bool,
        **kwargs,
    ):
        """
        Load tokenizer from HuggingFace hub or local path.

        WHY: Centralises the loading logic and handles the two supported backends.
        For SentencePiece, we try to locate the .model file; for HF, we use AutoTokenizer.

        Assumptions:
        - For SentencePiece, `model_name` is either a direct path to a .model file
          or a directory containing exactly one .model file.
        - For HF, `model_name` can be any identifier that AutoTokenizer understands.

        Edge Cases:
        - If the directory contains multiple .model files, we pick the first one
          alphabetically – this is arbitrary but deterministic. Better to specify
          the exact file path.
        - If the required library is not installed, raise an ImportError with
          installation instructions.
        """
        if tokenizer_type == "sentencepiece":
            if not SENTENCEPIECE_AVAILABLE:
                raise ImportError("sentencepiece is not installed. Run: pip install sentencepiece")
            model_path = model_name
            if not os.path.exists(model_path):
                # Try to find .model file in directory
                if os.path.isdir(model_path):
                    model_files = list(Path(model_path).glob("*.model"))
                    if model_files:
                        model_path = str(model_files[0])
                    else:
                        raise FileNotFoundError(f"No .model file found in {model_path}")
                else:
                    raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")
            self._tokenizer = spm.SentencePieceProcessor()
            # Load the model – Load() returns True on success, but we don't check.
            # If loading fails, it will raise an exception internally.
            self._tokenizer.Load(model_path)
            self._type = "sentencepiece"
            return

        # Default: HuggingFace tokenizer
        if not HF_AVAILABLE:
            raise ImportError("transformers is not installed. Run: pip install transformers")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self._cache_dir,
            use_fast=use_fast,
            **kwargs,
        )
        self._type = "hf"

    def _infer_special_tokens(self):
        """
        Set pad_token_id, eos_token_id, bos_token_id from the underlying tokenizer.

        WHY: For causal language modelling, we need a pad_token_id (for padding) and
        eos_token_id (for end-of-sequence). Many GPT‑style models do not have a pad_token
        by default. This method ensures that a valid pad_token_id is always set, falling
        back to eos_token_id or 0.

        Edge Cases:
        - If the tokenizer has no pad_token and also no eos_token, we set pad_token_id = 0.
          This is safe because 0 is usually the <unk> token, and training will still work
          (the model will learn to ignore padding if we set ignore_index appropriately).
        - For SentencePiece, we map the standard special tokens by their string names.
          If a token does not exist, we use a sensible default (0 for pad, 1 for eos).

        Side Effects: Modifies the internal cached attributes. Does NOT modify the
        underlying tokenizer's pad_token (unless you later call `add_special_tokens`).
        """
        if self._type == "hf":
            self._pad_token_id = self._tokenizer.pad_token_id
            self._eos_token_id = self._tokenizer.eos_token_id
            self._bos_token_id = self._tokenizer.bos_token_id
            self._vocab_size = self._tokenizer.vocab_size

            # If pad_token_id is None, set it to eos_token_id or 0
            if self._pad_token_id is None:
                if self._eos_token_id is not None:
                    self._pad_token_id = self._eos_token_id
                    warnings.warn(
                        f"Tokenizer has no pad_token. Setting pad_token_id = eos_token_id ({self._eos_token_id}).",
                        UserWarning,
                    )
                else:
                    self._pad_token_id = 0
                    warnings.warn(
                        "Tokenizer has no pad_token and no eos_token. Setting pad_token_id = 0.",
                        UserWarning,
                    )

        elif self._type == "sentencepiece":
            # SentencePiece uses <unk>, <s>, </s>, <pad> by convention.
            # piece_to_id returns -1 if the piece does not exist.
            pad_id = self._tokenizer.piece_to_id("<pad>")
            self._pad_token_id = pad_id if pad_id != -1 else 0
            eos_id = self._tokenizer.piece_to_id("</s>")
            self._eos_token_id = eos_id if eos_id != -1 else 1
            bos_id = self._tokenizer.piece_to_id("<s>")
            self._bos_token_id = bos_id if bos_id != -1 else None
            self._vocab_size = self._tokenizer.GetPieceSize()

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # -----------------------------------------------------------------------
    # Core tokenization methods
    # -----------------------------------------------------------------------

    def tokenize(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Convert a string to a list of token IDs.

        WHY: This is the primary method for turning raw text into token indices.
        It delegates to the underlying tokenizer's encoding method.

        Edge Cases:
        - Empty string returns [] (empty list) to avoid unnecessary padding tokens.
        - For HF fast tokenizers, `.encode` returns an `Encoding` object with `.ids`.
          We handle both the list and object cases.

        Performance: This method is intended for single‑example use (e.g., during
        interactive usage or small‑scale processing). For batch processing, use
        `batch_tokenize` which is optimised for many examples.

        Parameters
        ----------
        text : str
            Input text.
        add_special_tokens : bool
            Whether to add BOS/EOS tokens (if supported by the tokenizer).
        **kwargs
            Additional arguments passed to the underlying tokenizer (e.g., truncation, max_length).

        Returns
        -------
        List[int]
            Token IDs.
        """
        if not text:
            return []

        if self._type == "hf":
            out = self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
            # For fast tokenizers, encode returns a list; for slow, it's an encoding object
            if hasattr(out, "ids"):
                return out.ids
            return out
        elif self._type == "sentencepiece":
            # EncodeAsIds returns a list of integers directly.
            return self._tokenizer.EncodeAsIds(text)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self._type}")

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.

        WHY: Useful for debugging, inspecting model outputs, or reconstructing text
        after tokenization. Also needed for evaluation metrics like BLEU/ROUGE.

        Edge Cases:
        - Empty list returns empty string.
        - For HF tokenizers, `skip_special_tokens=True` removes tokens like <|endoftext|>.
          For SentencePiece, special tokens are retained unless you post‑process.

        Performance: Decoding is generally fast; used occasionally during evaluation.
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not token_ids:
            return ""
        if self._type == "hf":
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        elif self._type == "sentencepiece":
            return self._tokenizer.DecodeIds(token_ids)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self._type}")

    def encode(self, text: str, add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Backwards-compatible alias for ``tokenize``.

        Generation code in this repository historically called ``encode`` on
        Hugging Face tokenizers directly. Exposing the same method here keeps the
        generation stack decoupled from the underlying tokenizer implementation.
        """
        return self.tokenize(text, add_special_tokens=add_special_tokens, **kwargs)

    def batch_decode(
        self,
        sequences: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Backwards-compatible batch decoding interface.

        Hugging Face tokenizers expose ``batch_decode``; generation uses that API
        when converting sampled token IDs back into text. This wrapper keeps the
        same call pattern while routing everything through ``decode``.
        """
        if hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __call__(self, *args, **kwargs):
        """
        Make the wrapper behave like a tokenizer callable.

        This is intentionally a thin pass-through to ``batch_tokenize`` so existing
        generation code can keep using ``tokenizer(...)`` without depending on the
        underlying library.
        """
        return self.batch_tokenize(*args, **kwargs)

    def batch_tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Tokenize a batch of texts efficiently.

        WHY: Tokenising many examples one by one (calling `tokenize` in a loop) is
        slow because each call has Python overhead and may not utilise parallelism.
        This method leverages the underlying tokenizer's built‑in batch processing
        (for HF) or implements a manual loop with padding (for SentencePiece).

        For HuggingFace tokenizers:
            Uses the `__call__` method which is highly optimised (parallel tokenisation,
            vectorised padding, support for attention masks and special tokens).

        For SentencePiece:
            Falls back to a loop over `tokenize`, then manually applies padding and
            truncation. This is not as fast but acceptable for moderate batch sizes.

        Performance: For HF, this is O(batch_size) with low constant overhead. For
        SentencePiece, O(batch_size * seq_len). Prefer to use HF tokenizers when possible.

        Parameters
        ----------
        texts : List[str]
            List of input strings.
        add_special_tokens : bool
            Add BOS/EOS tokens.
        padding : bool or str
            Padding strategy: True/'max_length' (pad to max_length) or 'longest'
            (pad to longest in batch).
        truncation : bool or str
            Truncation strategy: True/'longest_first' or 'only_first'.
        max_length : int, optional
            Maximum length for padding/truncation. Required if padding='max_length'.
        return_tensors : str, optional
            If 'pt', return PyTorch tensors; if 'np', return numpy arrays.
        **kwargs
            Extra arguments (passed to HF tokenizer if applicable).

        Returns
        -------
        dict
            A dictionary containing at least 'input_ids'. May also include 'attention_mask'
            if the underlying tokenizer supports it or if we generate it manually.
        """
        if self._type == "hf":
            # Fast batch processing – the HF tokenizer handles everything.
            encoded = self._tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                **kwargs,
            )
            return encoded
        else:
            # Fallback for SentencePiece: tokenize one by one and manually pad.
            # This implementation is simpler but less efficient.
            token_ids_list = [self.tokenize(t, add_special_tokens=add_special_tokens) for t in texts]

            # Apply truncation if requested.
            if max_length is None and truncation:
                # If truncation is True but no max_length given, use the longest sequence length.
                max_length = max(len(ids) for ids in token_ids_list) if token_ids_list else 0
            if truncation and max_length is not None:
                token_ids_list = [ids[:max_length] for ids in token_ids_list]

            # Apply padding if requested.
            if padding:
                # Determine target length
                if padding == "max_length" and max_length is not None:
                    target_len = max_length
                else:
                    target_len = max(len(ids) for ids in token_ids_list) if token_ids_list else 0
                # Pad with pad_token_id on the right (standard for causal LMs)
                padded = [ids + [self.pad_token_id] * (target_len - len(ids)) for ids in token_ids_list]
                # Create attention mask: 1 for real tokens, 0 for padding.
                attn_mask = [[1] * len(ids) + [0] * (target_len - len(ids)) for ids in token_ids_list]
                token_ids_list = padded
            else:
                # No padding: attention mask is all ones.
                attn_mask = [[1] * len(ids) for ids in token_ids_list]

            # Convert to requested tensor type.
            if return_tensors == "pt":
                import torch
                input_ids = torch.tensor(token_ids_list, dtype=torch.long)
                attention_mask = torch.tensor(attn_mask, dtype=torch.long)
            elif return_tensors == "np":
                input_ids = np.array(token_ids_list, dtype=np.int32)
                attention_mask = np.array(attn_mask, dtype=np.int32)
            else:
                input_ids = token_ids_list
                attention_mask = attn_mask

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    # -----------------------------------------------------------------------
    # Preprocessing utilities for dataset generation
    # -----------------------------------------------------------------------

    def encode_file(
        self,
        input_file: str,
        output_file: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        batch_size: int = 1000,
        verbose: bool = True,
    ) -> None:
        """
        Tokenize a text file line by line and write token IDs as JSON lines or raw binary.

        WHY: This is a convenience method to convert a raw text corpus into the format
        required by `PretrainingDataset` (flat binary memmap) or `StreamingDataset`
        (JSONL). It handles streaming, batching, and progress bars automatically.

        Output formats:
        - If `output_file` ends with .npy or .bin: writes a flat binary array of int32.
          This is ideal for `PretrainingDataset` with memmap.
        - Otherwise: writes one JSON object per line `{"input_ids": [1,2,3]}`.
          This can be read by `PretrainingDataset` in memory mode or by a custom loader.

        Performance: Processes the file in batches to avoid holding all tokenised
        sequences in memory. For binary output, it accumulates all token IDs in a list
        before writing; for very large files (>100GB), this may consume too much memory.
        In that case, use `StreamingDataset` directly on the original Parquet or JSONL.

        Edge Cases:
        - Empty lines are skipped (not tokenised) to avoid empty sequences.
        - If `max_length` is provided, sequences are truncated.
        - Overwrites `output_file` if it already exists (no warning) – be cautious.

        Parameters
        ----------
        input_file : str
            Path to input text file (one sample per line).
        output_file : str
            Output path. If ends with .npy/.bin, writes flat binary; else JSONL.
        add_special_tokens : bool
            Whether to add BOS/EOS to each line.
        max_length : Optional[int]
            Truncate each sequence to this length.
        batch_size : int
            Number of lines to process at a time.
        verbose : bool
            Show progress bar.
        """
        from tqdm.auto import tqdm

        # Determine output format based on file extension.
        is_binary = output_file.endswith(('.npy', '.bin'))

        # Count lines for progress bar (requires reading the file once).
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        all_token_ids = []  # only used if is_binary
        token_count = 0

        with open(input_file, 'r', encoding='utf-8') as f_in:
            pbar = tqdm(total=total_lines, desc="Tokenizing", disable=not verbose, unit="line")

            batch = []
            for line in f_in:
                line = line.strip()
                if line:
                    batch.append(line)
                if len(batch) >= batch_size:
                    # Tokenise the batch.
                    encoded = self.batch_tokenize(
                        batch,
                        add_special_tokens=add_special_tokens,
                        truncation=True,
                        max_length=max_length,
                        return_tensors=None,  # get lists, not tensors
                    )
                    for ids in encoded['input_ids']:
                        if is_binary:
                            all_token_ids.extend(ids)
                        else:
                            # Append to JSONL file (one object per line).
                            with open(output_file, 'a') as f_out:
                                f_out.write(json.dumps({"input_ids": ids}) + "\n")
                    if is_binary:
                        token_count += sum(len(ids) for ids in encoded['input_ids'])
                    batch = []
                    pbar.update(batch_size)
            # Process the last batch (if any).
            if batch:
                encoded = self.batch_tokenize(
                    batch,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_length,
                )
                if is_binary:
                    for ids in encoded['input_ids']:
                        all_token_ids.extend(ids)
                    token_count += sum(len(ids) for ids in encoded['input_ids'])
                else:
                    for ids in encoded['input_ids']:
                        with open(output_file, 'a') as f_out:
                            f_out.write(json.dumps({"input_ids": ids}) + "\n")
                pbar.update(len(batch))
            pbar.close()

        if is_binary:
            # Write as flat array for memmap (`.npy` format).
            arr = np.array(all_token_ids, dtype=np.int32)
            np.save(output_file, arr)
            if verbose:
                print(f"Saved {len(arr)} tokens to {output_file}")

    # -----------------------------------------------------------------------
    # Serialisation (save/load) for compatibility
    # -----------------------------------------------------------------------

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the tokenizer to a directory so it can be reloaded with `from_pretrained`.

        WHY: This method provides compatibility with HuggingFace's ecosystem and allows
        the tokenizer to be stored alongside a model checkpoint. It also enables
        easy reproduction of tokenisation settings across runs.

        For HF tokenizers: delegates to `save_pretrained`.
        For SentencePiece: saves the .model file and a custom `tokenizer_config.json`.

        Assumptions: The directory must be writable. Existing files will be overwritten.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._type == "hf":
            self._tokenizer.save_pretrained(save_dir)
        elif self._type == "sentencepiece":
            model_path = save_dir / "spm.model"
            self._tokenizer.Save(str(model_path))
            config = {
                "tokenizer_type": "sentencepiece",
                "model_file": "spm.model",
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "bos_token_id": self.bos_token_id,
                "vocab_size": self.vocab_size,
            }
            with open(save_dir / "tokenizer_config.json", "w") as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported tokenizer type for saving: {self._type}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "KilatTokenizer":
        """
        Load a tokenizer from a directory or HuggingFace hub.

        WHY: Provides a symmetrical interface to `save_pretrained`. Allows the same
        code to work with both HF and SentencePiece tokenizers as long as the directory
        contains the appropriate files.

        Example:
            >>> tok = KilatTokenizer.from_pretrained("./my_tokenizer")
        """
        return cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    # -----------------------------------------------------------------------
    # Special token management
    # -----------------------------------------------------------------------

    def add_special_tokens(self, special_tokens: List[str]) -> None:
        """
        Add new special tokens to the tokenizer.

        WHY: Sometimes you need to add custom control tokens (e.g., <|im_start|>, <|im_end|>)
        for chat templates or instruction tuning. This method adds them and updates
        the vocabulary size.

        Caveats:
        - For HF tokenizers, it uses `add_special_tokens` which correctly resizes the
          embedding layer if needed. The user must subsequently resize the model embeddings
          (e.g., `model.resize_token_embeddings(tokenizer.vocab_size)`).
        - For SentencePiece, adding tokens after loading is not supported (the model is
          immutable). A warning is emitted and the request is ignored. To add tokens to
          SentencePiece, you must retrain the tokenizer with the new tokens included.

        Edge Cases:
        - Duplicate tokens are handled by the underlying tokenizer (usually ignored).
        - If the tokenizer already has the token, nothing changes.
        """
        if self._type == "hf":
            self._tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self._vocab_size = self._tokenizer.vocab_size
        elif self._type == "sentencepiece":
            # SentencePiece does not support adding tokens after loading easily.
            warnings.warn("Adding special tokens to SentencePiece tokenizer is not supported; ignoring.")
        else:
            raise ValueError(f"Unsupported tokenizer type: {self._type}")

    # -----------------------------------------------------------------------
    # Magic methods
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"KilatTokenizer(type={self._type}, vocab_size={self.vocab_size})"

    def __len__(self) -> int:
        return int(self.vocab_size or 0)
