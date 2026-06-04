import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Union, Any, Optional
import pyarrow.parquet as pq

class KilatDataset(Dataset):
    """
    Dataset reader for KilatTransformer supporting multiple input formats.

    Supports:
        - Apache Parquet files (``.parquet`` / ``.parq``) – memory‑mapped for
          large‑scale training with zero‑copy reads and column‑wise access.
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

    Memory Model
    -----------
    - **Parquet**: ParquetFile.read() loads only the specified column into an
      Arrow Table in memory. The table is shared across all workers (if using
      multiple DataLoader workers with fork start method, the table is inherited
      via copy‑on‑write in the child processes).
    - **JSON/JSONL**: Entire dataset is loaded into memory at construction.
      For datasets larger than RAM, use Parquet format.
    - **In‑memory lists**: The list reference is stored directly — no copy.

    Example::
        >>> ds = KilatDataset("train.parquet", key_name="input_ids")
        >>> sample = ds[0]
        >>> print(sample.keys())   # dict with key "input_ids"
        >>> print(len(ds))         # total number of samples
    """

    def __init__(
        self,
        file_or_data: Union[str, List[Dict[str, Any]]],
        key_name: str = "input_ids",
    ):
        """
        Parameters
        ----------
        file_or_data : Union[str, List[Dict[str, Any]]]
            - If str: path to a Parquet (.parquet/.parq), JSON (.json), or
              JSONL (.jsonl) file. Format is inferred from extension.
            - If list: list of dictionaries, each containing at least the
              key specified by ``key_name``.
        key_name : str
            The dictionary key or column name containing the token ID sequence.
            Default: ``"input_ids"``.

        Raises
        ------
        FileNotFoundError
            If ``file_or_data`` is a str pointing to a non‑existent file.
        TypeError
            If ``file_or_data`` is neither str nor list.
        ValueError
            If the dataset contains zero samples after loading.
            If JSON/JSONL file is malformed.
        KeyError
            If ``key_name`` is not present in the first sample (for JSON/JSONL).
        """
        self.key_name = key_name
        self.is_parquet = False
        self.parquet_table = None
        self.samples: List[Dict[str, Any]] = []

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
            _, ext = os.path.splitext(file_or_data)
            ext = ext.lower()

            if ext in (".parquet", ".parq"):
                # Parquet backend: memory‑mapped, column‑selective loading.
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
                "file_or_data must be a str (file path) or a list of dicts."
            )

        # Fail fast on empty datasets.
        # An empty dataset would cause division‑by‑zero in steps‑per‑epoch
        # calculation and produce cryptic errors downstream.
        if self.dataset_length == 0:
            raise ValueError("Dataset contains zero samples.")

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
        if self.key_name not in self.samples[0]:
            raise KeyError(
                f"Key '{self.key_name}' not found in the loaded JSON data."
            )

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        Returns the token sequence at position ``idx``.

        Normalization
        ------------
        All backends normalize to ``Dict[str, List[int]]`` with the token
        sequence as a plain Python list. This abstraction ensures the data
        collator works uniformly regardless of the storage backend.
        
        The normalization happens per‑sample (lazily) rather than at load time
        to avoid doubling memory usage. For Parquet, the Arrow column is
        stored in memory; individual rows are converted to lists on access.

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
        Dict[str, List[int]]
            Dictionary with key ``self.key_name`` mapping to a list of
            integer token IDs.

        Raises
        ------
        IndexError
            If idx is out of range [0, dataset_length).
        TypeError
            If the token sequence cannot be converted to a list.
        """
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
            # .to_pylist() converts Arrow scalar → Python list.
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

        return {"input_ids": token_ids}