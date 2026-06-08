from __future__ import annotations
import glob
import json
import math
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _discover_parquet_files(path: str) -> List[str]:
    """
    Return a sorted list of all .parquet / .parq files under ``path``.

    WHY: Both ParquetDataset and StreamingDataset need to locate all shards
    in a directory. Sorting ensures deterministic order across runs when
    shuffle=False, and consistent worker file distribution.

    Edge Cases:
    - If path is a single file, returns a one-element list.
    - If directory contains no .parquet files, raises ValueError (fail fast)
      rather than silently returning an empty iterator.
    - Hidden files (starting with .) are ignored because glob does not match them.

    Performance: O(n) where n = number of files. Called once per dataset
    construction, acceptable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    files += sorted(glob.glob(os.path.join(path, "*.parq")))
    if not files:
        raise ValueError(f"No .parquet / .parq files found in: {path}")
    return files


def _to_list(token_ids: Any, idx: int) -> List[int]:
    """
    Normalise any token sequence type to a plain Python list.

    WHY: The dataset may return token IDs as list, torch.Tensor, numpy array,
    or even JAX array (if using interop). The collator expects standard Python
    lists for easy batching and further processing. This helper centralises
    the conversion and gives a clear error message when an unsupported type appears.

    Edge Cases:
    - Recursive conversion is NOT performed; only the top-level object is converted.
    - For tensors, .tolist() copies data to Python ints – overhead is acceptable
      because token sequences are small (e.g., 1024 ints) and this conversion
      happens once per sample, not per batch.

    Raises:
    - TypeError if token_ids is of an unsupported type.
    """
    if isinstance(token_ids, list):
        return token_ids
    if isinstance(token_ids, torch.Tensor):
        return token_ids.tolist()
    if hasattr(token_ids, "tolist"):          # numpy, jax, etc.
        return token_ids.tolist()
    raise TypeError(
        f"Unsupported token sequence type at index {idx}: {type(token_ids)}"
    )


# ---------------------------------------------------------------------------
# 1. PretrainingDataset  (map-style, random access)
# ---------------------------------------------------------------------------

class PretrainingDataset(Dataset):
    """
    Map-style dataset for language model pretraining.

    WHY MAP-STYLE?
    Random access by index (__getitem__) is essential for:
    - Shuffling with a fixed seed (reproducible epochs)
    - Using PyTorch's RandomSampler / DistributedSampler
    - Resuming training exactly at a specific sample

    Supported storage backends (all present a uniform dict interface):

    1. **NumPy memmap** (``.npy`` / ``.bin``)
       - Fastest random access. File is memory-mapped, pages loaded on demand.
       - Assumes a flat int32/int64 array of concatenated token IDs.
       - Sequences are sliced at `[idx * chunk_size : (idx+1) * chunk_size]`.
       - Zero-copy: OS page cache shared across DataLoader workers.
       - Ideal for large pretraining corpora that exceed RAM.

    2. **Apache Parquet** (``.parquet`` / directory)
       - Column-selective loading via Arrow. Fetch row `i` with `table.column(key)[i]`.
       - Suitable when data is already in Parquet shards (HF datasets, Spark) and
         the dataset fits in RAM (or you have enough RAM to hold the whole table).
       - Slower than memmap for random access because Arrow must decode each row.

    3. **JSON / JSONL / in-memory list**
       - Full in-memory load. Suitable for small datasets, evaluation sets,
         or programmatic construction.

    Trade-offs:
    - Memmap: fastest, but requires pre‑tokenisation into a flat binary file.
      Also discards the tail (tokens % chunk_size) to keep all chunks equal length.
    - Parquet: more flexible (supports multiple columns, row groups), but each
      __getitem__ involves a PyArrow conversion (row → Python list). For large
      datasets, memmap can be 2-3x faster.
    - In‑memory: simplest, but uses RAM for the whole dataset.

    Parameters
    ----------
    source : str | List[dict]
        Path to a file or directory, or a list of dicts.
    key_name : str
        Column / dict key containing token IDs. Default "input_ids".
    chunk_size : int
        Sequence length for memmap slicing. Ignored for other backends.
    dtype : np.dtype
        Numpy dtype for memmap. Must match the file's dtype.

    Example Usage
    -------------
        >>> ds = PretrainingDataset("tokens.npy", chunk_size=1024)
        >>> len(ds)          # total_tokens // chunk_size
        >>> ds[0]            # {"input_ids": [3, 17, 42, ...]}
    """

    def __init__(
        self,
        source: Union[str, List[Dict[str, Any]]],
        key_name: str = "input_ids",
        chunk_size: int = 1024,
        dtype: np.dtype = np.int32,
    ) -> None:
        self.key_name = key_name
        self.chunk_size = chunk_size

        self._backend: str          # "memmap" | "parquet" | "memory"
        self._memmap: Optional[np.ndarray] = None
        self._table: Optional[pa.Table] = None
        self._samples: Optional[List[Dict[str, Any]]] = None
        self._memory_is_dicts = False

        if isinstance(source, list):
            self._backend = "memory"
            self._samples = source
            self._length = len(source)
            self._memory_is_dicts = bool(source) and isinstance(source[0], dict)

        elif isinstance(source, str):
            if os.path.isdir(source):
                self._init_parquet(source)
            else:
                _, ext = os.path.splitext(source)
                ext = ext.lower()

                if ext in (".parquet", ".parq"):
                    self._init_parquet(source)
                elif ext in (".npy", ".bin"):
                    self._init_memmap(source, dtype)
                elif ext in (".jsonl", ".json"):
                    self._init_json(source, ext)
                elif ext == "":
                    # A path without extension is ambiguous. Prefer treating
                    # it as a parquet source only if it is an actual directory.
                    raise ValueError(
                        f"Unsupported path without extension: {source}. "
                        "Use a .parquet/.parq file, a directory of parquet files, "
                        "a .npy/.bin memmap, or a JSON/JSONL file."
                    )
                else:
                    # Fallback: try parquet file(s) only after explicit types.
                    self._init_parquet(source)
        else:
            raise TypeError("source must be a str path or list of dicts.")

        if self._length == 0:
            raise ValueError("Dataset is empty after loading.")

    # ── backend initialisers ──────────────────────────────────────────────

    def _init_memmap(self, path: str, dtype: np.dtype) -> None:
        """
        Memory-map a flat integer array and slice into fixed-size chunks.

        WHY: Memmap gives O(1) index → byte offset, no parsing overhead.
        The file is opened read-only; data pages are loaded on first access
        and cached by the OS.

        Edge Cases:
        - If the file is an .npy, we load via np.load(…, mmap_mode='r').
        - Otherwise treat as raw binary (e.g., .bin) via np.memmap.
        - Tokens that do not fit into a full chunk are silently discarded.
          This is intentional: training requires all examples to have the
          same length for efficient batching without padding. Discarding
          a small tail (at most chunk_size-1 tokens) is acceptable for
          large pretraining corpora.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memmap file not found: {path}")
        self._memmap = np.load(path, mmap_mode="r") if path.endswith(".npy") else \
                       np.memmap(path, dtype=dtype, mode="r")
        total_tokens = len(self._memmap)
        self._length = total_tokens // self.chunk_size
        self._backend = "memmap"
        if self._length == 0:
            raise ValueError(
                f"Memmap file has {total_tokens} tokens but chunk_size={self.chunk_size}. "
                "Not enough tokens for even one chunk."
            )

    def _init_parquet(self, path: str) -> None:
        """Load Parquet file(s) column-selectively into an Arrow Table.

        WHY: Parquet is columnar; reading only the required column reduces I/O.
        We load the entire column into an Arrow Table because random access
        by row index is efficient (Arrow supports O(1) row slicing).

        Note: This loads the whole column into RAM. For huge datasets that
        do not fit in memory, use StreamingDataset instead.
        """
        files = _discover_parquet_files(path)
        tables = [pq.read_table(f, columns=[self.key_name]) for f in files]
        self._table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self._length = self._table.num_rows
        self._backend = "parquet"

    def _init_json(self, path: str, ext: str) -> None:
        """Load JSON / JSONL into memory.

        WHY: JSON is human-readable but slow to parse and not memory-efficient.
        Only suitable for small datasets or evaluation.
        """
        samples: List[Dict[str, Any]] = []
        if ext == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSONL at line {i+1}: {e}") from e
        else:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must be a top-level list of dicts.")
            samples = data
        if samples and self.key_name not in samples[0]:
            raise KeyError(f"Key '{self.key_name}' not found in JSON data.")
        self._samples = samples
        self._length = len(samples)
        self._backend = "memory"
        self._memory_is_dicts = bool(samples) and isinstance(samples[0], dict)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length - 1}].")

        if self._backend == "memmap":
            start = idx * self.chunk_size
            # Slicing a memmap returns a view; .tolist() copies into Python ints.
            token_ids = self._memmap[start : start + self.chunk_size].tolist()

        elif self._backend == "parquet":
            # Arrow column access: .column(name)[idx] returns a PyArrow scalar,
            # .as_py() converts to Python list.
            token_ids = self._table.column(self.key_name)[idx].as_py()

        else:  # memory
            if self._memory_is_dicts:
                token_ids = self._samples[idx][self.key_name]
            else:
                token_ids = self._samples[idx]

        return {self.key_name: _to_list(token_ids, idx)}


# ---------------------------------------------------------------------------
# 2. StreamingDataset  (iterable, memory-efficient)
# ---------------------------------------------------------------------------

class StreamingDataset(IterableDataset):
    """
    Iterable dataset for corpora that do not fit in RAM.

    WHY ITERABLE?
    For very large datasets (hundreds of GB or more), random access is impossible
    because indexes would require an O(N) mapping. IterableDataset processes
    data sequentially, streaming from disk without building a global index.

    Memory usage is O(read_batch_size × source_sequence_length) independent of
    corpus size.

    DataLoader integration:
        loader = DataLoader(StreamingDataset(...), batch_size=8,
                            collate_fn=CLMCollator(pad_token_id=0))

    Worker sharding (automatic):
        When num_workers > 0, __iter__ detects worker id and assigns a disjoint
        subset of files to each worker via `_get_worker_files`. This prevents
        duplicate samples across workers. No custom worker_init_fn needed.

    Epoch shuffling:
        Set shuffle=True to randomise file order and row order within each read‑batch
        at the start of every epoch. Call `set_epoch(epoch)` before each epoch to
        advance the seed; otherwise the same order repeats.

    How attention_mask is handled:
        - If the Parquet schema includes an `attention_mask` column, it is used
          to determine valid token lengths (padding already marked).
        - Otherwise, valid length is inferred by stripping trailing pad_token_id
          values from the token sequence. This works for HuggingFace tokenised
          datasets where all sequences are right-padded.

    Parameters
    ----------
    source : str
        Path to a single Parquet file or a directory of Parquet files.
    key_name : str
        Column name containing token IDs.
    chunk_size : int
        Length of each output sequence chunk.
    pad_token_id : int
        Token ID used to identify padding (determines valid token count).
    shuffle : bool
        Shuffle file order and within-batch row order each epoch.
    drop_last : bool
        Discard the final incomplete chunk per sequence.
    seed : int
        Base random seed; actual seed = seed + epoch.
    read_batch_size : int
        Rows read per Parquet iteration step (larger = less I/O overhead,
        but more memory).

    Example Usage
    -------------
        ds = StreamingDataset("./data/train/", chunk_size=1024, shuffle=True)
        loader = DataLoader(ds, batch_size=16, num_workers=4,
                            collate_fn=CLMCollator(pad_token_id=0))
        for epoch in range(10):
            ds.set_epoch(epoch)
            for batch in loader:
                loss = model(**batch)
    """

    def __init__(
        self,
        source: str,
        key_name: str = "input_ids",
        chunk_size: int = 1024,
        pad_token_id: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        read_batch_size: int = 256,
    ) -> None:
        self.key_name = key_name
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.read_batch_size = read_batch_size

        self._files = _discover_parquet_files(source)
        self._epoch = 0

        # Detect presence of attention_mask column across the available files.
        # If any shard carries the mask, we can use it directly; otherwise we
        # infer padding by stripping pad_token_id.
        self._has_mask = False
        for file_path in self._files:
            schema = pq.read_schema(file_path)
            if any(name.strip().lower() == "attention_mask" for name in schema.names):
                self._has_mask = True
                break

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for deterministic shuffling.

        WHY: The dataset itself holds the seed and epochs; we need to advance
        the seed every epoch so that shuffle order differs. This method should
        be called before each epoch in the training loop.

        Edge Cases: If not called, self._epoch remains 0 and shuffling (if enabled)
        repeats the same order every epoch – which might be acceptable for
        deterministic evaluation but undesirable for training.

        Example:
            for epoch in range(num_epochs):
                dataset.set_epoch(epoch)
                for batch in loader: ...
        """
        self._epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        """
        Yield one chunk at a time as {key_name: List[int]}.

        This method is called once per DataLoader worker. It respects worker
        sharding automatically via `_get_worker_files`.
        """
        worker_info = torch.utils.data.get_worker_info()
        files = self._get_worker_files(worker_info)

        rng = random.Random(self.seed + self._epoch)
        if self.shuffle:
            rng.shuffle(files)

        for chunk in self._iter_chunks(files, rng):
            yield {self.key_name: chunk}
        # NOTE: Do NOT auto-increment _epoch here.
        # Each DataLoader worker holds an independent copy of the dataset.
        # Incrementing inside __iter__ would cause the epoch counter to drift
        # across workers. The user must call set_epoch explicitly in the main
        # process before each epoch.

    # ── internals ─────────────────────────────────────────────────────────

    def _get_worker_files(self, worker_info) -> List[str]:
        """Return the file subset assigned to this DataLoader worker."""
        if worker_info is None:
            return list(self._files)
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
        # Round‑robin distribution: worker i gets files[i::num_workers]
        # This ensures each worker processes a disjoint subset.
        return self._files[worker_id::num_workers]

    def _iter_chunks(
        self,
        files: List[str],
        rng: random.Random,
    ) -> Iterator[List[int]]:
        """
        Core generator: read Parquet in batches → extract valid tokens → split into chunks.

        Steps:
        1. Read a batch of `read_batch_size` rows.
        2. For each row, obtain token_ids and (if available) attention_mask.
        3. If mask not present, infer it by stripping trailing pad tokens.
        4. Shuffle the rows within the batch (if shuffle=True).
        5. For each row, cut the active token sequence into chunks of `chunk_size`.
        6. Yield each chunk.

        Memory: At most `read_batch_size` rows are held in memory at any time.
        """
        columns = [self.key_name] + (["attention_mask"] if self._has_mask else [])

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.read_batch_size, columns=columns):
                token_rows = batch.column(0).to_pylist()

                if self._has_mask:
                    mask_rows = batch.column(1).to_pylist()
                else:
                    # Infer valid length by stripping trailing pad tokens.
                    # This is O(len(row)) per row, but batch size is moderate.
                    mask_rows = [
                        self._infer_mask(row) for row in token_rows
                    ]

                pairs = list(zip(token_rows, mask_rows))
                if self.shuffle:
                    rng.shuffle(pairs)

                for token_ids, mask in pairs:
                    valid_len = int(sum(mask))
                    # Skip sequences that are too short to form even one chunk.
                    # This avoids yielding empty chunks that would cause training instability.
                    if valid_len < 2:
                        continue
                    active = token_ids[:valid_len]
                    for start in range(0, len(active) - 1, self.chunk_size):
                        chunk = active[start : start + self.chunk_size]
                        if len(chunk) < 2:
                            continue
                        if len(chunk) < self.chunk_size and self.drop_last:
                            continue
                        yield chunk

    def _infer_mask(self, token_ids: List[int]) -> List[int]:
        """Build a binary mask by stripping trailing pad tokens."""
        valid_len = len(token_ids)
        while valid_len > 0 and token_ids[valid_len - 1] == self.pad_token_id:
            valid_len -= 1
        return [1] * valid_len + [0] * (len(token_ids) - valid_len)


# ---------------------------------------------------------------------------
# 3. PackedDataset  (bin-packing, zero padding waste)
# ---------------------------------------------------------------------------

class PackedDataset(Dataset):
    """
    Map-style dataset that packs multiple short sequences into fixed-length
    blocks with no padding waste.

    MOTIVATION
    Standard batching pads all sequences in a batch to the longest sequence,
    wasting compute on pad tokens. For pretraining on long documents, this can
    waste 20–40% of tokens. Bin‑packing concatenates sequences end‑to‑end
    (separated by EOS) to fill blocks exactly, achieving ~100% token utilisation.

    ALGORITHM (greedy first‑fit)
    At construction time:
        1. Maintain a current block buffer.
        2. For each sequence (with EOS appended):
            a. If it fits in the current block, append it.
            b. Otherwise, pad the current block to block_size (only the last
               block needs padding; intermediate blocks are exact).
            c. Start a new block with the current sequence.
        3. Drop or keep the last partial block according to `drop_last`.

    Why greedy first‑fit? It is simple, deterministic, and packs nearly as
    well as more complex algorithms for typical sequence length distributions.

    ATTENTION MASKING
    Each packed block carries a corresponding `labels` tensor where padding
    positions are set to -100 (ignored by nn.CrossEntropyLoss). This ensures:
    - Loss is computed only over real tokens.
    - No cross‑contamination between sequences: the EOS token teaches the model
      to end a sequence, and the next token starts a new context.

    For models that need to prevent attention from crossing document boundaries
    (e.g., flash‑attention with cu_seqlens), the `block_seqlens` list stores
    the lengths of individual sequences within each block. This can be passed
    to the model's attention function.

    MEMORY TRADE‑OFF
    PackedDataset is map‑style, so it must build the packed blocks at construction
    time. This requires materialising all source sequences into a list of token
    lists. For very large corpora (e.g., >100GB), use StreamingDataset directly
    with a collator that pads (accepting some waste) or implement a streaming
    packer as a custom IterableDataset.

    Parameters
    ----------
    source : PretrainingDataset | StreamingDataset | List[List[int]]
        Source of token sequences. For StreamingDataset, all data is materialised
        into memory at construction time – only suitable for moderate‑sized datasets.
    block_size : int
        Target packed block length (= model's max_position_embeddings).
    eos_token_id : int
        Appended after each sequence before packing.
    pad_token_id : int
        Fills the tail of the last (partial) block.
    drop_last : bool
        If True, discard the last block if it is shorter than block_size.

    Example Usage
    -------------
        base = PretrainingDataset("tokens.npy", chunk_size=256)
        packed = PackedDataset(base, block_size=1024, eos_token_id=2)
        len(packed)        # number of full blocks
        packed[0]          # dict with keys: input_ids, labels, block_seqlens
    """

    def __init__(
        self,
        source: Union[PretrainingDataset, "StreamingDataset", List[List[int]]],
        block_size: int = 1024,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        sequences = self._collect_sequences(source)
        self._blocks, self._seqlens = self._pack(sequences, drop_last)

    # ── sequence collection ───────────────────────────────────────────────

    def _collect_sequences(
        self,
        source: Any,
    ) -> List[List[int]]:
        """
        Extract raw token sequences from any supported source type.

        WHY: PackedDataset must know all sequences upfront to run the packing
        algorithm. For StreamingDataset, this means materialising the entire
        iterable into a list. This is intentional: PackedDataset is a map‑style
        dataset; it requires a fixed length. If the corpus is too large to
        materialise, use StreamingDataset directly with a collator that pads,
        or implement a custom streaming packer.
        """
        if isinstance(source, list):
            # List of token sequences or list of dicts
            if source and isinstance(source[0], dict):
                # List[Dict[str, List[int]]] – extract the first key
                key = next(iter(source[0]))
                return [_to_list(s[key], i) for i, s in enumerate(source)]
            return [_to_list(s, i) for i, s in enumerate(source)]

        if isinstance(source, StreamingDataset):
            # Materialise the iterable – only appropriate for moderate‑size datasets
            return [item[source.key_name] for item in source]

        if isinstance(source, PretrainingDataset):
            key = source.key_name
            return [source[i][key] for i in range(len(source))]

        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Use PretrainingDataset, StreamingDataset, or List[List[int]]."
        )

    # ── greedy first‑fit packing ──────────────────────────────────────────

    def _pack(
        self,
        sequences: List[List[int]],
        drop_last: bool,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Greedily pack sequences into blocks of block_size tokens.

        Returns
        -------
        blocks : List[List[int]]
            Each entry is a token ID list of length <= block_size.
            Intermediate blocks have length exactly block_size (after padding);
            the last block may be shorter unless drop_last discards it.
        seqlens : List[List[int]]
            Each entry is a list of individual sequence lengths packed
            into the corresponding block (used for cu_seqlens / flash‑attn).

        Edge Cases:
        - If a single sequence (after adding EOS) is longer than block_size,
          we truncate it to block_size - 1 and force a final EOS. This is safer
          than splitting into multiple blocks, because splitting would require
          placing EOS at each sub‑block boundary to maintain document separation.
          Truncation is acceptable in well‑preprocessed corpora where tokenizer
          already limits sequence length.
        """
        blocks: List[List[int]] = []
        seqlens: List[List[int]] = []

        current_block: List[int] = []
        current_seqlens: List[int] = []

        for seq in sequences:
            # Append EOS to delimit sequences within a block
            tokens = seq + [self.eos_token_id]

            # Handle overly long sequences
            if len(tokens) > self.block_size:
                tokens = tokens[: self.block_size - 1] + [self.eos_token_id]
                # Now length = block_size, fall through to normal packing.

            # Check if tokens fit in current block
            if len(current_block) + len(tokens) <= self.block_size:
                current_seqlens.append(len(tokens))
                current_block.extend(tokens)
            else:
                # Flush current block and start a new one
                if current_block:
                    blocks.append(self._pad_block(current_block))
                    seqlens.append(current_seqlens)
                current_block = list(tokens)
                current_seqlens = [len(tokens)]

        # Handle the last partial block
        if current_block:
            if not drop_last:
                blocks.append(self._pad_block(current_block))
                seqlens.append(current_seqlens)

        return blocks, seqlens

    def _pad_block(self, block: List[int]) -> List[int]:
        """Right‑pad block to block_size with pad_token_id."""
        pad_len = self.block_size - len(block)
        return block + [self.pad_token_id] * pad_len

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a packed block as a training‑ready dict.

        Keys:
        - input_ids : List[int] of length block_size
        - labels    : List[int] where pad positions are -100 (ignored by loss)
        - block_seqlens : List[int] of individual sequence lengths inside the block
        """
        if idx < 0 or idx >= len(self._blocks):
            raise IndexError(f"Index {idx} out of range [0, {len(self._blocks) - 1}].")

        block = self._blocks[idx]
        labels = [
            tok if tok != self.pad_token_id else -100
            for tok in block
        ]
        return {
            "input_ids": block,
            "labels": labels,
            "block_seqlens": self._seqlens[idx],
        }


# ---------------------------------------------------------------------------
# 4. ConcatDataset  (weighted mix dari multiple sources)
# ---------------------------------------------------------------------------

class ConcatDataset(Dataset):
    """
    Mix multiple map‑style datasets with optional sampling weights.

    WHY: Pretraining corpora typically blend multiple sources (web text, books,
    code, multilingual) at specific mixing ratios. ConcatDataset implements
    two strategies:

    **Proportional mode (weights=None)**
        Datasets are concatenated sequentially. Total length = sum of lengths.
        Equivalent to simple concatenation. Index mapping uses binary search
        over cumulative sizes (O(log n)).

    **Weighted mode (weights provided)**
        A virtual dataset of total_size samples is constructed by drawing
        samples from source datasets according to the weights (normalised).
        The mapping is pre‑computed at construction so __getitem__ is O(1)
        and deterministic (no RNG at access time).

        This is the standard approach used by The Pile, RedPajama, and Dolma
        for mixing heterogeneous pretraining corpora.

    Important restriction: All source datasets must be map‑style (have __len__).
    IterableDataset (StreamingDataset) is not supported because ConcatDataset
    needs to know the length of each source to build the index map.

    Parameters
    ----------
    datasets : List[Dataset]
        Source datasets. All must implement __len__ and __getitem__.
    weights : Optional[List[float]]
        Sampling weight for each dataset. Automatically normalised to sum=1.
        If None, datasets are concatenated without resampling.
    total_size : Optional[int]
        Target virtual dataset size for weighted mode. Defaults to the sum
        of all source dataset lengths.
    seed : int
        Random seed for constructing the weighted index map.

    Example Usage
    -------------
        web = PretrainingDataset("web_tokens.npy", chunk_size=1024)
        books = PretrainingDataset("book_tokens.npy", chunk_size=1024)
        code = PretrainingDataset("code_tokens.npy", chunk_size=1024)

        # Equal concatenation
        ds = ConcatDataset([web, books, code])

        # Weighted: 70% web, 20% books, 10% code
        ds = ConcatDataset([web, books, code], weights=[0.7, 0.2, 0.1])
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        total_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if not datasets:
            raise ValueError("datasets list is empty.")
        # Verify all datasets are map‑style (have __len__)
        for i, ds in enumerate(datasets):
            if not hasattr(ds, "__len__"):
                raise TypeError(
                    f"ConcatDataset only supports map‑style datasets with __len__. "
                    f"datasets[{i}] is {type(ds).__name__} (IterableDataset). "
                    "For streaming mixing, instantiate one StreamingDataset per source "
                    "and interleave them in the training loop instead."
                )
        if weights is not None and len(weights) != len(datasets):
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(datasets)={len(datasets)}."
            )

        self.datasets = datasets

        if weights is None:
            self._weighted = False
            self._cumulative_sizes = self._compute_cumulative_sizes()
            self._length = self._cumulative_sizes[-1]
        else:
            self._weighted = True
            total = total_size or sum(len(d) for d in datasets)
            self._length = total
            self._index_map = self._build_index_map(weights, total, seed)

    # ── index mapping ─────────────────────────────────────────────────────

    def _compute_cumulative_sizes(self) -> List[int]:
        """Cumulative sizes for O(log n) binary‑search indexing."""
        sizes: List[int] = []
        cumsum = 0
        for d in self.datasets:
            cumsum += len(d)
            sizes.append(cumsum)
        return sizes

    def _build_index_map(
        self,
        weights: List[float],
        total: int,
        seed: int,
    ) -> List[Tuple[int, int]]:
        """
        Pre‑compute a mapping from virtual index → (dataset_idx, local_idx).

        Steps:
        1. Normalise weights to sum 1.
        2. Allocate sample counts per dataset (rounding, last adjusted).
        3. For each dataset, assign local indices by cycling (local_idx % len(ds)).
        4. Shuffle the combined list to interleave sources throughout the epoch.
        """
        # Normalise weights
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # Compute exact sample count per dataset
        counts = [round(w * total) for w in norm_weights]
        # Fix rounding error: adjust the last bucket
        counts[-1] += total - sum(counts)

        # Build unshuffled index map
        index_map: List[Tuple[int, int]] = []
        for ds_idx, (ds, count) in enumerate(zip(self.datasets, counts)):
            ds_len = len(ds)
            for sample_num in range(count):
                local_idx = sample_num % ds_len  # cycle if count > ds_len
                index_map.append((ds_idx, local_idx))

        # Shuffle to interleave sources
        rng = random.Random(seed)
        rng.shuffle(index_map)
        return index_map

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Any:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length - 1}].")

        if self._weighted:
            ds_idx, local_idx = self._index_map[idx]
            return self.datasets[ds_idx][local_idx]

        # Proportional mode: binary search into cumulative sizes
        ds_idx = self._find_dataset_index(idx)
        local_idx = idx - (self._cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0)
        return self.datasets[ds_idx][local_idx]

    def _find_dataset_index(self, idx: int) -> int:
        """Binary search: find which dataset owns global index idx."""
        lo, hi = 0, len(self.datasets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._cumulative_sizes[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        return lo


# ---------------------------------------------------------------------------
# Legacy alias — backward compatibility with existing code using KilatDataset
# ---------------------------------------------------------------------------

class KilatDataset(PretrainingDataset):
    """
    Backward‑compatible alias for PretrainingDataset.

    .. deprecated::
        Use PretrainingDataset for map‑style access or StreamingDataset
        for iterable / large‑corpus usage. KilatDataset will be removed
        in a future release.

    Example (legacy):
        ds = KilatDataset("tokens.bin", key_name="input_ids")
    """

    def __init__(
        self,
        file_or_data: Union[str, List[Dict[str, Any]]],
        key_name: str = "input_ids",
        # Legacy streaming param — ignored; use StreamingDataset instead
        streaming: bool = False,
        **kwargs,
    ) -> None:
        import warnings
        warnings.warn(
            "KilatDataset is deprecated. Use PretrainingDataset (map‑style) "
            "or StreamingDataset (iterable) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if streaming:
            raise ValueError(
                "streaming=True is no longer supported in KilatDataset. "
                "Use StreamingDataset(source, chunk_size=...) instead."
            )
        super().__init__(source=file_or_data, key_name=key_name)
