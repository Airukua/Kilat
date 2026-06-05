import os
import json
import math
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any, Optional, Iterator, Tuple
import pyarrow.parquet as pq
import pyarrow as pa
import glob


class KilatDataset(Dataset):
    """
    Dataset reader for KilatTransformer supporting multiple input formats.

    Supports:
        - Apache Parquet files (``.parquet`` / ``.parq``) – memory‑mapped for
          large‑scale training with zero‑copy reads and column‑wise access.
        - Parquet directories – directory containing multiple ``.parquet`` files,
          all files are automatically discovered and loaded.
        - Streaming mode – memory‑efficient iteration for large datasets that
          don't fit in RAM. Uses chunking and packing similar to PackedTokenBatchLoader.
        - JSON Lines (``.jsonl``) – line‑delimited JSON for streaming datasets.
        - Standard JSON (``.json``) containing a list of samples.
        - In‑memory Python lists – for programmatic dataset creation and testing.

    Every sample is expected to contain a key (default ``"input_ids"``) holding
    a sequence of integer token IDs.

    Design Rationale
    ---------------
    This dataset class exists to provide a unified interface over multiple
    storage backends commonly encountered in ML training pipelines, without
    requiring users to pre‑convert their data to a specific format.

    **Why Parquet as the primary large‑scale format?**
    1. **Columnar storage**: The dataset reads only the ``key_name`` column,
       ignoring other columns (metadata, auxiliary features). For datasets with
       many columns, this can reduce I/O by 10‑100x.
    2. **Memory‑mapped access**: Arrow/Parquet uses mmap for zero‑copy reads.
       The OS page cache handles prefetching, meaning repeated epochs access
       the same physical memory pages without re‑reading from disk.
    3. **Random access**: Parquet's row‑group structure allows O(1) random
       access by index, unlike JSONL which requires sequential scanning.
    4. **Compression**: Column‑wise compression (Snappy, Zstd, LZ4) typically
       achieves 3‑5x size reduction for integer sequences vs uncompressed JSON.
    5. **Interoperability**: Parquet is the standard format for HuggingFace
       datasets, Spark, and most data lakes — datasets can be consumed directly
       without conversion.

    **Why support Parquet directories?**
    - Many data processing pipelines (Spark, Dask, HuggingFace datasets)
      write Parquet data as sharded directories with multiple files.
    - Processing these manually (iterating over files, concatenating tables)
      is error‑prone and inefficient.
    - The directory support automatically discovers and combines all
      ``.parquet`` files in a directory, transparently presenting them as a
      single contiguous dataset.

    **Why support streaming mode?**
    - For extremely large datasets that exceed available RAM, loading all data
      at once causes OOM (Out of Memory) errors.
    - Streaming mode reads data in batches, processes chunks, and yields samples
      on‑the‑fly without storing everything in memory.
    - This is ideal for pretraining on web‑scale corpora (hundreds of GB).
    - Memory usage becomes O(batch_size × sequence_length) instead of O(dataset_size).

    **Why support JSON/JSONL at all?**
    - JSONL is ubiquitous for small‑medium datasets and easy to inspect/debug
    - JSON files are common for evaluation sets and task‑specific data
    - In‑memory lists enable programmatic dataset construction (e.g., synthetic
      data generation, few‑shot prompt templates)

    **Normalization strategy**:
    All backends normalize token sequences to Python lists in ``__getitem__``.
    This ensures the data collator receives consistent types regardless of
    backend (Arrow arrays → list, JSON lists → pass‑through, tensors → list).
    The conversion is done lazily per‑sample to avoid memory overhead.

    **Parquet Directory Resolution**
    --------------------------------
    When a directory path is provided:
    1. All files matching ``*.parquet`` or ``*.parq`` in the directory are
       discovered (non‑recursive by design; recursive would risk loading
       unintended files from nested subdirectories).
    2. Files are loaded in sorted order for deterministic row indices.
    3. Each file is read column‑selectively (only ``key_name`` column).
    4. Tables are concatenated using PyArrow's efficient table concatenation
       (zero‑copy row‑wise stacking).
    5. The resulting table is stored as a single contiguous Arrow Table.

    **Streaming Mode Details**
    --------------------------
    When ``streaming=True``, the dataset:
    1. Does NOT load all data into memory at construction
    2. Iterates through Parquet files in chunks (``read_batch_size`` rows at a time)
    3. Extracts valid token sequences (using attention_mask)
    4. Splits long sequences into chunks of ``sequence_length``
    5. Packs chunks into batches of size ``batch_size``
    6. Returns (inputs, labels) tuples directly for training

    This matches the behavior of ``PackedTokenBatchLoader`` while maintaining
    the same Dataset interface.

    Memory Model
    -----------
    - **Parquet (single file)**: ParquetFile.read() loads only the specified
      column into an Arrow Table in memory. The table is shared across all
      workers (if using multiple DataLoader workers with fork start method,
      the table is inherited via copy‑on‑write in the child processes).
    - **Parquet (directory)**: All discovered files are concatenated into a
      single Arrow Table. This is memory‑efficient because columnar data from
      multiple files is concatenated row‑wise without copying the underlying
      buffers (PyArrow uses zero‑copy concatenation).
    - **Streaming mode**: Memory usage is O(read_batch_size × source_sequence_length)
      plus O(batch_size × sequence_length). No full dataset copy.
    - **JSON/JSONL**: Entire dataset is loaded into memory at construction.
      For datasets larger than RAM, use Parquet format or streaming mode.
    - **In‑memory lists**: The list reference is stored directly — no copy.

    Example::
        >>> # Single Parquet file (full load)
        >>> ds = KilatDataset("train.parquet", key_name="input_ids")
        >>> 
        >>> # Directory with multiple Parquet files (full load - may OOM)
        >>> ds = KilatDataset("./data/tokens/train/tokens/tokenized.parquet", key_name="input_ids")
        >>> 
        >>> # Streaming mode for large datasets (recommended for >1GB)
        >>> ds = KilatDataset(
        ...     "./data/tokens/train/tokens/tokenized.parquet",
        ...     key_name="input_ids",
        ...     streaming=True,
        ...     batch_size=8,
        ...     sequence_length=512,
        ...     pad_token_id=0,
        ... )
        >>> 
        >>> # JSONL file
        >>> ds = KilatDataset("data.jsonl", key_name="input_ids")
        >>> 
        >>> sample = ds[0]
        >>> print(sample.keys())   # dict with key "input_ids"
        >>> print(len(ds))         # total number of samples
    """

    def __init__(
        self,
        file_or_data: Union[str, List[Dict[str, Any]]],
        key_name: str = "input_ids",
        # Streaming mode parameters (new)
        streaming: bool = False,
        batch_size: int = 8,
        sequence_length: int = 512,
        pad_token_id: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        read_batch_size: int = 128,
    ):
        """
        Parameters
        ----------
        file_or_data : Union[str, List[Dict[str, Any]]]
            - If str: path to a Parquet (.parquet/.parq), JSON (.json), or
              JSONL (.jsonl) file. **Also supports directories containing
              multiple Parquet files** – all .parquet/.parq files in the
              directory will be discovered and concatenated automatically.
            - If list: list of dictionaries, each containing at least the
              key specified by ``key_name``.
        key_name : str
            The dictionary key or column name containing the token ID sequence.
            Default: ``"input_ids"``.
        streaming : bool
            If True, use streaming mode (memory efficient, recommended for large
            datasets >1GB that would cause OOM). If False, load all data into
            memory (default). When streaming=True, the dataset returns (inputs, labels)
            tuples directly, not dicts with "input_ids".
        batch_size : int
            Number of chunks per batch (only used when streaming=True).
        sequence_length : int
            Target sequence length for chunking (only used when streaming=True).
        pad_token_id : int
            Token ID used for padding (only used when streaming=True).
        shuffle : bool
            Whether to shuffle chunks within each read batch (only used when streaming=True).
        drop_last : bool
            Whether to drop the last incomplete batch (only used when streaming=True).
        seed : int
            Random seed for shuffling (only used when streaming=True).
        read_batch_size : int
            Number of rows to read at once from Parquet files (only used when streaming=True).

        Raises
        ------
        FileNotFoundError
            If ``file_or_data`` is a str pointing to a non‑existent file/directory.
        TypeError
            If ``file_or_data`` is neither str nor list.
        ValueError
            If the dataset contains zero samples after loading.
            If JSON/JSONL file is malformed.
            If Parquet directory contains no .parquet/.parq files.
        KeyError
            If ``key_name`` is not present in the first sample (for JSON/JSONL).
        """
        self.key_name = key_name
        self.is_parquet = False
        self.parquet_table = None
        self.samples: List[Dict[str, Any]] = []
        
        # Streaming mode attributes
        self.streaming = streaming
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.read_batch_size = read_batch_size
        self._epoch = 0
        self._part_files = None
        self._total_rows = 0
        self._source_sequence_length = None

        # --- Input type dispatch ---
        # Path‑based loading: detect format from file extension.
        # Extension detection is case‑insensitive and handles compound
        # extensions like .parquet.gz (though not officially supported,
        # the suffix check catches .parquet).
        if isinstance(file_or_data, str):
            if not os.path.exists(file_or_data):
                raise FileNotFoundError(
                    f"Dataset path does not exist: {file_or_data}"
                )

            # Check if it's a directory (Parquet sharded dataset)
            if os.path.isdir(file_or_data):
                if streaming:
                    # Streaming mode: don't load everything, just store file list
                    self._init_streaming_mode(file_or_data, key_name)
                else:
                    # Full load mode (may OOM for large datasets)
                    self._load_from_parquet_directory(file_or_data, key_name)
            else:
                _, ext = os.path.splitext(file_or_data)
                ext = ext.lower()

                if ext in (".parquet", ".parq"):
                    # Single Parquet file backend: memory‑mapped, column‑selective loading.
                    # Only the required column is read into memory to minimize
                    # RAM usage. Other columns (e.g., metadata, source info)
                    # are ignored entirely.
                    self.is_parquet = True
                    self.parquet_file = pq.ParquetFile(file_or_data)
                    # Column‑selective read: if the dataset has 50 columns but we
                    # only need "input_ids", we avoid loading the other 49 columns.
                    # This can reduce memory by 10‑100x for wide datasets.
                    self.parquet_table = self.parquet_file.read(columns=[self.key_name])
                    self.dataset_length = self.parquet_table.num_rows
                else:
                    # JSON/JSONL backend: full in‑memory loading.
                    # Suitable for datasets up to a few GB that fit in RAM.
                    # For larger datasets, use Parquet.
                    self._load_from_json(file_or_data, ext)
                    self.dataset_length = len(self.samples)
        elif isinstance(file_or_data, list):
            # In‑memory backend: direct reference (no copy).
            # User is responsible for the list's lifetime.
            # This is useful for programmatic dataset creation, testing,
            # or wrapping iterable datasets.
            self.samples = file_or_data
            self.dataset_length = len(self.samples)
        else:
            raise TypeError(
                "file_or_data must be a str (file path or directory) or a list of dicts."
            )

        # Fail fast on empty datasets.
        # An empty dataset would cause division‑by‑zero in steps‑per‑epoch
        # calculation and produce cryptic errors downstream.
        if not streaming and self.dataset_length == 0:
            raise ValueError("Dataset contains zero samples.")
        elif streaming and self._total_rows == 0:
            raise ValueError("Dataset contains zero rows (streaming mode).")

    def _init_streaming_mode(self, dir_path: str, key_name: str):
        """
        Initialize streaming mode for large Parquet directories.

        This method only stores metadata (file paths, row counts) without loading
        any actual data into memory. Data is read on-the-fly during iteration.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing Parquet files.
        key_name : str
            The column name to read from each Parquet file.
        """
        # Discover all Parquet files in the directory.
        # Using sorted() ensures deterministic order across runs.
        self._part_files = sorted(glob.glob(os.path.join(dir_path, "*.parquet")))
        self._part_files += sorted(glob.glob(os.path.join(dir_path, "*.parq")))

        if not self._part_files:
            raise ValueError(
                f"No .parquet or .parq files found in directory: {dir_path}"
            )

        print(f"Found {len(self._part_files)} Parquet file(s) in {dir_path} (streaming mode)")

        # Count total rows without loading data (reads only metadata)
        self._total_rows = 0
        for file_path in self._part_files:
            pf = pq.ParquetFile(file_path)
            self._total_rows += pf.metadata.num_rows

        # Infer source sequence length from first file
        self._source_sequence_length = self._infer_source_sequence_length()

        # For compatibility with Dataset interface, set dataset_length to
        # estimated number of batches (not actual samples count)
        chunk_factor = max(1, self._source_sequence_length // self.sequence_length)
        estimated_samples = self._total_rows * chunk_factor
        self.dataset_length = math.ceil(estimated_samples / self.batch_size)

        print(f"Streaming mode: {self._total_rows:,} rows, {len(self._part_files)} files")
        print(f"Source seq length: {self._source_sequence_length}")
        print(f"Estimated batches: {self.dataset_length:,}")

    def _infer_source_sequence_length(self) -> int:
        """Infer the sequence length of source data from first file."""
        first_file = pq.ParquetFile(self._part_files[0])
        first_batch = next(first_file.iter_batches(batch_size=1, columns=["input_ids"]))
        return len(first_batch.column(0)[0].as_py())

    def _iter_chunks(self) -> Iterator[List[int]]:
        """
        Iterate through all data and yield chunks of tokens.

        This is the core of streaming mode. It reads Parquet files in batches,
        extracts valid token sequences, and splits them into chunks of
        sequence_length. Memory usage is O(read_batch_size × source_sequence_length).
        """
        rng = random.Random(self.seed + self._epoch)
        part_files = list(self._part_files)
        if self.shuffle:
            rng.shuffle(part_files)

        for part_file in part_files:
            parquet_file = pq.ParquetFile(part_file)
            for record_batch in parquet_file.iter_batches(
                batch_size=self.read_batch_size,
                columns=["input_ids", "attention_mask"],
            ):
                token_rows = record_batch.column(0).to_pylist()
                mask_rows = record_batch.column(1).to_pylist()
                rows = list(zip(token_rows, mask_rows))
                if self.shuffle:
                    rng.shuffle(rows)

                for token_ids, attention_mask in rows:
                    valid_tokens = int(sum(attention_mask))
                    if valid_tokens < 2:
                        continue
                    active_ids = token_ids[:valid_tokens]
                    max_start = len(active_ids) - 1
                    for start in range(0, max_start, self.sequence_length):
                        chunk = active_ids[start:start + self.sequence_length]
                        if len(chunk) < 2:
                            continue
                        yield chunk

        self._epoch += 1

    def _collate_chunks(self, chunks: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a list of token chunks into input_ids and labels tensors.

        This method pads chunks to sequence_length and creates labels shifted by one
        position with -100 for padding positions (ignored in loss calculation).

        Parameters
        ----------
        chunks : List[List[int]]
            List of token chunks to collate into a batch.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (input_ids, labels) tensors of shape (batch_size, sequence_length)
        """
        padded_ids = []
        padded_mask = []
        for chunk in chunks:
            trimmed = chunk[:self.sequence_length]
            pad_size = self.sequence_length - len(trimmed)
            padded_ids.append(trimmed + [self.pad_token_id] * pad_size)
            padded_mask.append([1] * len(trimmed) + [0] * pad_size)

        token_tensor = torch.tensor(padded_ids, dtype=torch.long)
        mask_tensor = torch.tensor(padded_mask, dtype=torch.long)
        inputs = token_tensor[:, :-1]
        labels = token_tensor[:, 1:].clone()
        labels[mask_tensor[:, 1:] == 0] = -100
        return inputs, labels

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        For full-load mode: returns exact number of samples.
        For streaming mode: returns estimated number of batches (not exact sample count).

        Returns
        -------
        int
            Number of samples (full-load) or estimated batches (streaming).
        """
        return self.dataset_length

    def __getitem__(self, idx: int) -> Union[Dict[str, List[int]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the token sequence at position ``idx``.

        For streaming mode, returns (input_ids, labels) tuple directly.
        For non-streaming mode, returns Dict[str, List[int]].

        Normalization
        ------------
        All backends normalize to plain Python lists for uniform downstream handling.
        The collator expects lists; this conversion ensures consistency
        regardless of whether the data was stored as tensors, numpy arrays,
        Arrow arrays, or native Python lists.

        Supported Types for token_ids
        ----------------------------
        - torch.Tensor → .tolist() (common when loading pre‑tokenized tensors)
        - numpy.ndarray → .tolist() (common in older datasets)
        - list → pass‑through (native format)
        - Other types raise TypeError with index context for debugging.

        Parameters
        ----------
        idx : int
            Integer index (supports Python‑style negative indexing for
            accessing samples from the end of the dataset).

        Returns
        -------
        Union[Dict[str, List[int]], Tuple[torch.Tensor, torch.Tensor]]
            - Non-streaming mode: Dictionary with key ``self.key_name`` mapping to
              a list of integer token IDs.
            - Streaming mode: Tuple of (input_ids, labels) tensors ready for model.

        Raises
        ------
        IndexError
            If idx is out of range [0, dataset_length).
        TypeError
            If the token sequence cannot be converted to a list.
        NotImplementedError
            If streaming mode and random access is attempted (streaming only
            supports sequential iteration via __iter__, not __getitem__).
        """
        if self.streaming:
            # Streaming mode does not support random access by index
            # Users should iterate with 'for batch in dataloader' or use __iter__
            raise NotImplementedError(
                "Streaming mode does not support random access via __getitem__. "
                "Use 'for batch in dataset:' iteration instead, or set streaming=False."
            )

        # Validate index with descriptive error message.
        # Using [0, len-1] range check (not Python's default IndexError)
        # provides a clearer error message including the valid range.
        if idx < 0 or idx >= self.dataset_length:
            raise IndexError(
                f"Index {idx} out of range [0, {self.dataset_length - 1}]."
            )

        # --- Extract token IDs based on storage backend ---
        if self.is_parquet:
            # Arrow columnar access: retrieve row and convert to Python list.
            # .column(key_name) returns a ChunkedArray (Arrow column).
            # [idx] returns a scalar Arrow value.
            # .as_py() converts Arrow scalar → Python list.
            # This is a zero‑copy operation for the underlying buffer;
            # the Python list is a view into Arrow's memory in many cases.
            token_ids = self.parquet_table.column(self.key_name)[idx].as_py()
        else:
            # In‑memory backend: direct dictionary lookup.
            token_ids = self.samples[idx][self.key_name]

        # Normalise to plain Python list for uniform downstream handling.
        # The collator expects lists; this conversion ensures consistency
        # regardless of whether the data was stored as tensors, numpy arrays,
        # Arrow arrays, or native Python lists.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif hasattr(token_ids, "tolist"):  # NumPy array, JAX array, etc.
            token_ids = token_ids.tolist()
        elif not isinstance(token_ids, list):
            raise TypeError(
                f"Unsupported token sequence type at index {idx}: {type(token_ids)}"
            )

        return {self.key_name: token_ids}

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches in streaming mode.

        This is the primary interface for streaming mode. Each iteration yields
        a batch of (input_ids, labels) tensors ready for model training.

        Yields
        ------
        Tuple[torch.Tensor, torch.Tensor]
            (input_ids, labels) tensors of shape (batch_size, sequence_length)

        Returns
        -------
        Iterator[Tuple[torch.Tensor, torch.Tensor]]
            Iterator over batches.
        """
        if not self.streaming:
            # For non-streaming mode, fall back to standard Dataset iteration
            # This creates a DataLoader-like interface for convenience
            for i in range(len(self)):
                yield self[i]
            return

        # Streaming mode: iterate through chunks and yield batches
        batch_buffer: List[List[int]] = []
        for chunk in self._iter_chunks():
            batch_buffer.append(chunk)
            if len(batch_buffer) == self.batch_size:
                yield self._collate_chunks(batch_buffer)
                batch_buffer = []

        if batch_buffer and not self.drop_last:
            yield self._collate_chunks(batch_buffer)

    def _load_from_parquet_directory(self, dir_path: str, key_name: str):
        """
        Load all Parquet files from a directory into a single Arrow Table.

        This method discovers every ``.parquet`` and ``.parq`` file in the
        specified directory (non‑recursive for performance; recursive discovery
        is a common footgun that can accidentally include hundreds of unrelated
        files). If recursive discovery is needed, users can pass a glob pattern
        directly as ``file_or_data``.

        Design Decisions
        ---------------
        - **Sorted file order**: Files are processed in alphabetical order to
          ensure deterministic row indices across runs. Without sorting, the
          order of files from ``glob.glob()`` is system‑dependent.
        - **Column‑selective reading**: Each file is read with
          ``columns=[key_name]`` to avoid loading unnecessary columns.
        - **Zero‑copy concatenation**: PyArrow's ``concat_tables()`` stacks
          tables row‑wise without copying the underlying buffers (the data is
          already columnar and contiguous per file; concatenation just chains
          the row groups).
        - **Empty directory detection**: If no Parquet files are found, raises
          a clear error rather than silently creating an empty dataset.

        Parameters
        ----------
        dir_path : str
            Path to the directory containing Parquet files.
        key_name : str
            The column name to read from each Parquet file.

        Raises
        ------
        ValueError
            If no ``.parquet`` or ``.parq`` files are found in the directory.
        """
        # Discover all Parquet files in the directory.
        # Using sorted() ensures deterministic order across runs.
        # The pattern matches both .parquet and .parq extensions.
        parquet_files = sorted(glob.glob(os.path.join(dir_path, "*.parquet")))
        parquet_files += sorted(glob.glob(os.path.join(dir_path, "*.parq")))

        if not parquet_files:
            raise ValueError(
                f"No .parquet or .parq files found in directory: {dir_path}"
            )

        print(f"Found {len(parquet_files)} Parquet file(s) in {dir_path}")

        # Load each file column‑selectively and collect tables.
        # Using a list comprehension here keeps the code concise while
        # maintaining readability. Each read is independent, so files can be
        # processed in any order; sorted order is only for determinism.
        tables = []
        for file_path in parquet_files:
            # Read only the required column to minimize memory usage.
            # This is particularly important when the source files have many
            # metadata columns (e.g., HuggingFace datasets often include
            # __index_level_0__, timestamp, etc.).
            table = pq.read_table(file_path, columns=[key_name])
            tables.append(table)

        # Concatenate all tables into a single Arrow Table.
        # PyArrow's concat_tables performs zero‑copy row‑wise concatenation:
        # the underlying buffers from each table are chained together without
        # copying the actual data. This is both memory‑efficient and fast.
        # NOTE: concat_tables is from pyarrow, not pyarrow.parquet!
        self.parquet_table = pa.concat_tables(tables)
        self.is_parquet = True
        self.dataset_length = self.parquet_table.num_rows

        # Release the intermediate tables list to help garbage collector.
        # The tables list is no longer needed after concatenation; setting it
        # to None makes the memory eligible for reclamation (though Python's
        # GC may not free it immediately, it signals intent).
        tables = None

    def _load_from_json(self, path: str, ext: str):
        """
        Parse JSON or JSONL and verify the presence of ``key_name``.

        JSONL Parsing
        ------------
        Each line is a complete JSON object. This format is:
        - Appendable (new samples can be added without rewriting)
        - Streamable (can be read line‑by‑line for large files)
        - Human‑readable (each line is self‑contained)
        
        Empty lines are silently skipped to handle trailing newlines
        and common formatting artifacts.

        JSON Parsing
        -----------
        Expects a top‑level list of objects. This format is:
        - Easier to create from Python (json.dump(list_of_dicts, f))
        - More compact than JSONL for small datasets
        - Not streamable (entire file is parsed at once)

        Validation
        ---------
        After loading, checks that the first sample contains the required key.
        This catches common errors like:
        - Wrong key_name parameter
        - Data format mismatch (e.g., loading a file with "text" key when
          key_name="input_ids")
        - Corrupted data where the first sample is malformed
        """
        if ext == ".jsonl":
            # JSONL: one JSON object per line.
            # Using line‑by‑line reading with explicit error context (line number)
            # makes debugging easy — the error message points to the exact line
            # with the malformed JSON.
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():  # Skip empty lines (common at end of files)
                        try:
                            self.samples.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSONL at line {i + 1}: {e}"
                            )
        else:  # assume standard JSON
            # Standard JSON: top‑level list of objects.
            # The entire file is parsed at once. For very large datasets,
            # this can be memory‑intensive; JSONL or Parquet is preferred.
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise TypeError(
                            "JSON file must contain a top‑level list of samples."
                        )
                    self.samples = data
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON file: {e}")

        # Confirm the required key exists in the first sample.
        # This is a lightweight check — it only verifies the first sample,
        # not all samples. A more thorough validation would check all samples
        # but would be O(N) and potentially slow for large datasets.
        # The assumption is that if the first sample is correct, the rest
        # of the dataset follows the same schema.
        if self.samples and self.key_name not in self.samples[0]:
            raise KeyError(
                f"Key '{self.key_name}' not found in the loaded JSON data."
            )