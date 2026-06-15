from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import yaml
from .base import dump_yaml_file, load_yaml_file


class DataLoaderConfig:
    """
    DataLoader configuration for optimal data loading performance.

    WHY THIS EXISTS:
        DataLoader settings are often scattered across training scripts, leading to
        inconsistent configurations between training and evaluation. This class
        centralises all data loading parameters, validates them at construction,
        and provides YAML serialisation for reproducible experimentation.

    DESIGN DECISIONS:
        - **Worker management**: `num_workers` controls CPU cores used for data
          loading. `persistent_workers=True` keeps processes alive across epochs,
          reducing startup overhead for large datasets.
        - **Memory transfer**: `pin_memory=True` enables page-locked memory,
          significantly speeding up host-to-GPU transfers (2-3x for large batches).
        - **Prefetching**: `prefetch_factor` preloads batches to overlap I/O with
          computation. Higher values improve throughput but increase memory usage.
        - **Packing**: `use_packing` enables bin-packing of short sequences into
          fixed-size blocks, eliminating padding waste (improves token utilisation
          from ~70% to near 100% for variable-length sequences).
        - **Distributed**: `use_distributed_sampler=True` automatically partitions
          data across GPUs. Set `distributed_shuffle=False` for deterministic
          evaluation ordering.
        - **Truncation**: `"right"` removes tokens from the end (preserves prefix),
          `"left"` removes from the start (preserves suffix for generation tasks).

    PERFORMANCE NOTES:
        - For CPU-only training, `num_workers` should be set to number of CPU cores.
        - For GPU training, `num_workers=4` to `8` is typically optimal.
        - `persistent_workers=True` adds ~1-2 seconds to first epoch but saves
          ~0.5-1 second per subsequent epoch.
        - `prefetch_factor=2` is the PyTorch default; values up to 4 can improve
          throughput on NVMe drives.

    EDGE CASES:
        - `packed_block_size` defaults to `max_seq_length` if not provided.
        - `dataset_format="memmap"` expects a flat `.npy` file with concatenated tokens.
        - `dataset_format="parquet"` can read either a single file or directory of files.
        - `dataset_format="jsonl"` expects one JSON object per line with `input_ids` key.
        - If `train_data_path` is None, the trainer will use the dataset passed directly.
        - `prefetch_factor` is ignored when `num_workers=0` (PyTorch behaviour).

    Example Usage
    -------------
        >>> dl_config = DataLoaderConfig(
        ...     train_batch_size=32,
        ...     eval_batch_size=64,
        ...     num_workers=8,
        ...     max_seq_length=2048,
        ...     use_packing=True,
        ... )
        >>> dl_config.to_yaml("dataloader_config.yaml")
        >>> loaded = DataLoaderConfig.from_yaml("dataloader_config.yaml")
    """

    def __init__(
        self,
        # ---- Core settings ----
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        drop_last: bool = True,
        
        # ---- Sequence handling ----
        max_seq_length: int = 1024,
        truncation: Literal["left", "right"] = "right",
        
        # ---- Packing (optional) ----
        use_packing: bool = False,
        packed_block_size: Optional[int] = None,
        
        # ---- Distributed ----
        use_distributed_sampler: bool = True,
        distributed_shuffle: bool = True,
        
        # ---- Dataset source ----
        train_data_path: Optional[str] = None,
        eval_data_path: Optional[str] = None,
        dataset_format: Literal["parquet", "memmap", "jsonl"] = "parquet",
        
        # ---- Caching ----
        cache_dir: Optional[str] = None,
        prefetch_batches: int = 2,
    ):
        """
        Initialise DataLoader configuration with validation.

        Parameters
        ----------
        train_batch_size : int
            Training batch size per device (micro-batch before accumulation).
            Must be >= 1.
        eval_batch_size : int
            Evaluation batch size per device. Must be >= 1.
        num_workers : int
            Number of CPU subprocesses for data loading. 0 = main process only.
            Negative values are rejected.
        pin_memory : bool
            If True, use pinned memory for faster host-to-GPU transfers.
        prefetch_factor : int
            Batches to prefetch per worker (higher = more memory, better I/O overlap).
            Must be >= 1. Ignored when num_workers=0.
        persistent_workers : bool
            Keep worker processes alive across epochs. Reduces startup overhead.
            Ignored when num_workers=0.
        drop_last : bool
            Discard last incomplete batch (True for training to avoid variable sizes).
        max_seq_length : int
            Maximum sequence length after truncation. Must be >= 1.
        truncation : Literal["left", "right"]
            Which side to truncate. "right" keeps prefix (standard for LM training),
            "left" keeps suffix (useful for generation tasks).
        use_packing : bool
            Enable bin-packing of short sequences into fixed blocks. Reduces padding
            waste from 20-40% to near 0% for variable-length datasets.
        packed_block_size : Optional[int]
            Block size for packing. Defaults to max_seq_length if None.
        use_distributed_sampler : bool
            If True and distributed training, use DistributedSampler for data
            partitioning. Set to False only for custom sampling strategies.
        distributed_shuffle : bool
            If True, DistributedSampler shuffles data each epoch. Set to False
            for deterministic evaluation ordering.
        train_data_path : Optional[str]
            Path to training dataset (overrides hardcoded paths). If None, the
            trainer expects a dataset to be passed directly.
        eval_data_path : Optional[str]
            Path to evaluation dataset.
        dataset_format : Literal["parquet", "memmap", "jsonl"]
            Storage format for automatic DataLoader creation:
            - "parquet": Apache Parquet (supports directories or single files)
            - "memmap": Flat `.npy` binary file (fastest random access)
            - "jsonl": JSON lines (one JSON object per line)
        cache_dir : Optional[str]
            Directory for caching processed datasets (e.g., tokenized Parquet).
        prefetch_batches : int
            Number of batches to prefetch in dataset iterator. Larger values
            improve throughput at cost of memory.
        """
        # Validation (fail fast with clear error messages)
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        if prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
        if max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {max_seq_length}")
        if train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {train_batch_size}")
        if eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
        if prefetch_batches < 1:
            raise ValueError(f"prefetch_batches must be >= 1, got {prefetch_batches}")

        # Core settings
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        
        # Sequence handling
        self.max_seq_length = max_seq_length
        self.truncation = truncation
        
        # Packing (block_size defaults to max_seq_length)
        self.use_packing = use_packing
        self.packed_block_size = packed_block_size if packed_block_size is not None else max_seq_length
        
        # Distributed
        self.use_distributed_sampler = use_distributed_sampler
        self.distributed_shuffle = distributed_shuffle
        
        # Dataset source
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.dataset_format = dataset_format
        
        # Caching
        self.cache_dir = cache_dir
        self.prefetch_batches = prefetch_batches

    def to_dict(self) -> dict:
        """
        Convert DataLoader configuration to a plain dictionary.

        Returns
        -------
        dict
            Dictionary with all DataLoader hyperparameters, suitable for
            JSON/YAML serialisation or for passing to function signatures.
        """
        return {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
            "max_seq_length": self.max_seq_length,
            "truncation": self.truncation,
            "use_packing": self.use_packing,
            "packed_block_size": self.packed_block_size,
            "use_distributed_sampler": self.use_distributed_sampler,
            "distributed_shuffle": self.distributed_shuffle,
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            "dataset_format": self.dataset_format,
            "cache_dir": self.cache_dir,
            "prefetch_batches": self.prefetch_batches,
        }

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        """
        Export DataLoader configuration to YAML format.

        WHY YAML: Human-readable, supports comments (unlike JSON), and is the
        standard format for configuration files in the ML ecosystem.

        Parameters
        ----------
        path : Optional[str | Path]
            If provided, writes YAML to this file. If None, returns the YAML string.

        Returns
        -------
        str
            YAML string representation.
        """
        if path is not None:
            return dump_yaml_file(self.to_dict(), path)
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,  # Block style for readability
            sort_keys=False,           # Preserve logical parameter grouping
            allow_unicode=True,
            width=120,                 # Wide lines for better density
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> DataLoaderConfig:
        """
        Load DataLoader configuration from a YAML file.

        Enables workflow:
            1. Edit dataloader_config.yaml with desired parameters
            2. Load: config = DataLoaderConfig.from_yaml("config.yaml")
            3. Use in trainer: trainer = KilatTrainer(..., dataloader_config=config)

        Parameters
        ----------
        yaml_path : str | Path
            Path to YAML DataLoader configuration file.

        Returns
        -------
        DataLoaderConfig
            Validated DataLoader configuration instance.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        yaml.YAMLError
            If the YAML file is malformed.
        ValueError
            If the loaded configuration contains invalid values.
        """
        config_dict = load_yaml_file(yaml_path)
        return cls(**config_dict)

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"DataLoaderConfig(train_batch_size={self.train_batch_size}, "
            f"eval_batch_size={self.eval_batch_size}, "
            f"num_workers={self.num_workers}, "
            f"max_seq_length={self.max_seq_length}, "
            f"use_packing={self.use_packing})"
        )

    def get_effective_batch_size(self, gradient_accumulation_steps: int = 1) -> int:
        """
        Compute effective global batch size accounting for gradient accumulation.

        WHY: The effective batch size is train_batch_size * gradient_accumulation_steps
        (times world_size in distributed mode). This helper provides a single source
        of truth for logging and validation.

        Parameters
        ----------
        gradient_accumulation_steps : int
            Number of micro-batches before optimizer step.

        Returns
        -------
        int
            Effective batch size per GPU (not multiplied by world_size).
        """
        return self.train_batch_size * gradient_accumulation_steps