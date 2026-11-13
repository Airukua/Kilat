from __future__ import annotations
import logging
import random
from typing import Any, Callable, Optional
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from .collator import KilatDataCollator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    """Return True if torch.distributed is available and initialised."""
    # Layer 2 (data) must not import Layer 3 (engine): query torch.distributed directly.
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Return the number of processes in the distributed group (1 if not distributed).

    WHY: Used to configure DistributedSampler and to scale batch sizes etc.
    In DDP, world_size is the number of GPUs.
    """
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    """
    Return the rank of the current process (0 if not distributed).

    WHY: Rank 0 is typically used for logging, saving checkpoints, and acting
    as the master process in distributed training.
    """
    return dist.get_rank() if is_distributed() else 0


# ---------------------------------------------------------------------------
# Worker initialisation (reproducibility)
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """
    Initialise random seeds for each DataLoader worker independently.

    WHY: Each worker process must have a different seed so that random
    operations (e.g., data augmentation, shuffling within a dataset)
    produce different results across workers. Without this, all workers
    would generate the same sequence of random numbers, leading to
    duplicated samples and poor diversity.

    Assumptions:
    - The DataLoader's `worker_init_fn` calls this function with the worker ID.
    - The base seed is provided by the user (or default) and we add worker_id
      to create unique seeds.

    Edge Cases:
    - If num_workers = 0 (no separate processes), this function is never called.
    - The seeds are set for Python's random, NumPy, and PyTorch (CPU/CUDA).
      This does not guarantee full reproducibility across CUDA operations
      unless deterministic algorithms are enabled separately.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# DataLoader factory (core)
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    shuffle: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    persistent_workers: bool = False,
    seed: int = 42,
    use_distributed_sampler: bool = True,
    **kwargs: Any,
) -> DataLoader:
    distributed_shuffle: bool = kwargs.pop("distributed_shuffle", True)
    distributed_seed: Optional[int] = kwargs.pop("distributed_seed", None)
    """
    Factory to create a DataLoader with optimal configuration (including DDP).

    Parameter priority:
        - If `sampler` is provided, it is used directly (ignores shuffle, distributed logic).
        - If distributed and `use_distributed_sampler=True`, create a DistributedSampler.
        - Otherwise, create RandomSampler (if shuffle=True) or SequentialSampler.

    WHY: Centralises all DataLoader creation logic. Handles the subtle differences
    between single‑GPU and multi‑GPU training, worker initialisation, and collation.
    This ensures that all DataLoaders are created consistently across the codebase.

    Important design decisions:
        - Default collate_fn uses KilatDataCollator with pad_token_id=0 (common default).
          This is a convenience; for production, users should provide an explicit
          collate_fn to avoid ambiguity.
        - When distributed and use_distributed_sampler=True, we set `shuffle=False`
          because the DistributedSampler already handles shuffling.
        - `prefetch_factor` and `persistent_workers` are only used if num_workers>0.

    Edge cases:
        - If the dataset is an IterableDataset, sampler must be None and shuffle
          must be False. This function does not enforce it; the user must pass
          correct parameters (PyTorch will error otherwise).
        - If collate_fn is None and KilatDataCollator is not available (import error),
          the function will fail. The user should always provide a collate_fn.

    Performance:
        - prefetch_factor=2 (default) preloads 2 batches per worker to overlap I/O.
        - pin_memory=True speeds up host-to-GPU transfer for page‑locked memory.
        - persistent_workers keeps workers alive across epochs, reducing process
          creation overhead (especially useful for large datasets).

    Returns
    -------
    DataLoader
        Configured DataLoader ready for training or evaluation.
    """
    # If no collate_fn provided, fall back to KilatDataCollator with a common default.
    # This is only a convenience; the user is strongly encouraged to pass an explicit
    # collate_fn to avoid hidden assumptions about pad_token_id, etc.
    if collate_fn is None:
        logger.warning("No collate_fn provided, using KilatDataCollator with pad_token_id=0")
        collate_fn = KilatDataCollator(pad_token_id=0)

    # Sampler selection – user-provided sampler takes precedence.
    if sampler is None:
        if is_distributed() and use_distributed_sampler:
            # DistributedSampler: each process sees a unique subset of the data.
            # It can shuffle the order globally if distributed_shuffle=True.
            sampler = DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=distributed_shuffle,
                seed=distributed_seed if distributed_seed is not None else seed,
                drop_last=drop_last,
            )
            # We set shuffle=False because the DistributedSampler already handles ordering.
            shuffle = False
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            shuffle = False

    # Top-level picklable worker init function for reproducibility across spawn/fork processes.
    seeded_worker_init = _SeededWorkerInit(seed)

    # Golden Invariant #5: Only enable page-locked memory when CUDA accelerator is present
    effective_pin_memory = bool(pin_memory and torch.cuda.is_available())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        drop_last=drop_last,
        worker_init_fn=seeded_worker_init if num_workers > 0 else None,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        **kwargs,
    )

    logger.info(
        "Built DataLoader | batch_size=%d | num_workers=%d | pin_memory=%s | drop_last=%s | distributed=%s",
        batch_size,
        num_workers,
        pin_memory,
        drop_last,
        is_distributed(),
    )
    return dataloader


class _SeededWorkerInit:
    """Top-level picklable worker seeding callable (safe under Python `spawn` / `forkserver`)."""

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def __call__(self, worker_id: int) -> None:
        worker_init_fn(worker_id, base_seed=self.base_seed)

def build_train_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    ignore_index: int = -100,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    persistent_workers: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    use_distributed_sampler: bool = kwargs.pop("use_distributed_sampler", True)
    """
    Build a DataLoader specifically for training with sensible defaults.

    WHY: Training requires shuffling, usually drop_last=True to avoid
    variable batch sizes, and uses a DistributedSampler with shuffle enabled.
    This wrapper provides a simpler interface for the most common training case.

    If `collate_fn` is not provided, it creates a KilatDataCollator using the
    given pad_token_id, max_length, and ignore_index. This is the same collator
    used in the KilatTrainer for causal language modelling.

    Parameters
    ----------
    dataset : Dataset
        Training dataset (map-style).
    batch_size : int
        Batch size per GPU.
    collate_fn : Optional[Callable]
        Custom collation function. If None, uses KilatDataCollator.
    pad_token_id : int
        Token ID for padding (used by default collator).
    max_length : Optional[int]
        Maximum sequence length; longer sequences are truncated.
    ignore_index : int
        Value to ignore in loss computation (default -100).
    shuffle : bool
        Shuffle the dataset each epoch.
    num_workers : int
        Number of subprocesses for data loading.
    pin_memory : bool
        If True, use pinned memory for faster GPU transfer.
    prefetch_factor : int
        Batches to prefetch per worker.
    drop_last : bool
        Discard the last incomplete batch (default True for training).
    persistent_workers : bool
        Keep worker processes alive across epochs.
    seed : int
        Base seed for worker initialisation.
    use_distributed_sampler : bool
        If True and distributed, use DistributedSampler.
    **kwargs
        Additional arguments passed to `build_dataloader`.

    Returns
    -------
    DataLoader
        Configured training DataLoader.
    """
    if collate_fn is None:
        collate_fn = KilatDataCollator(
            pad_token_id=pad_token_id,
            max_length=max_length,
            ignore_index=ignore_index,
        )

    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        seed=seed,
        use_distributed_sampler=use_distributed_sampler,
        distributed_shuffle=True,   # training typically shuffles each epoch
        **kwargs,
    )


def build_eval_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    ignore_index: int = -100,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
    use_distributed_sampler: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Build a DataLoader specifically for evaluation with sensible defaults.

    WHY: Evaluation does not require shuffling, and we usually want to see
    results on all samples (drop_last=False). The DistributedSampler is
    still used to partition data across GPUs, but shuffling is disabled
    to ensure deterministic ordering.

    Parameters
    ----------
    dataset : Dataset
        Evaluation dataset (map-style).
    batch_size : int
        Batch size per GPU.
    collate_fn : Optional[Callable]
        Custom collation function. If None, uses KilatDataCollator.
    pad_token_id : int
        Token ID for padding (used by default collator).
    max_length : Optional[int]
        Maximum sequence length; longer sequences are truncated.
    ignore_index : int
        Value to ignore in loss computation (default -100).
    num_workers : int
        Number of subprocesses for data loading.
    pin_memory : bool
        If True, use pinned memory for faster GPU transfer.
    prefetch_factor : int
        Batches to prefetch per worker.
    drop_last : bool
        Discard the last incomplete batch (default False for evaluation).
    persistent_workers : bool
        Keep worker processes alive across epochs.
    seed : int
        Base seed for worker initialisation.
    use_distributed_sampler : bool
        If True and distributed, use DistributedSampler (with shuffle=False).
    **kwargs
        Additional arguments passed to `build_dataloader`.

    Returns
    -------
    DataLoader
        Configured evaluation DataLoader.
    """
    if collate_fn is None:
        collate_fn = KilatDataCollator(
            pad_token_id=pad_token_id,
            max_length=max_length,
            ignore_index=ignore_index,
        )

    return build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,                     # no shuffling for evaluation
        sampler=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        seed=seed,
        use_distributed_sampler=use_distributed_sampler,
        distributed_shuffle=False,         # deterministic order across ranks
        **kwargs,
    )


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    """
    Set the epoch for any DistributedSampler inside the DataLoader.

    WHY: DistributedSampler uses the epoch number to shuffle the data
    differently each epoch. If this is not called, the same order is
    repeated every epoch, reducing randomness and possibly hurting
    convergence. Must be called at the beginning of each training epoch.

    The function checks both `sampler` and `batch_sampler` (some DataLoaders
    use a custom batch sampler) for a `set_epoch` method.

    Parameters
    ----------
    dataloader : DataLoader
        The DataLoader (possibly containing a DistributedSampler).
    epoch : int
        Current epoch number (0-indexed).

    Example
    -------
        for epoch in range(num_epochs):
            set_dataloader_epoch(train_loader, epoch)
            for batch in train_loader:
                ...
    """
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    # Some DataLoaders use a batch_sampler instead (e.g., when using
    # a custom batch sampler). Check and set epoch there too.
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def build_dataloaders_from_config(
    main_config: Any,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build ``(train_dataloader, eval_dataloader)`` directly from a ``MainConfig``.
    Supports ``dataset_format="synthetic"`` for zero-dependency offline execution
    as well as ``parquet``, ``memmap``, and ``jsonl`` corpora.
    """
    from .dataset import PackedDataset, PretrainingDataset, SyntheticTokenDataset

    dl_cfg = main_config.dataloader
    model_cfg = main_config.model
    train_cfg = main_config.training

    collator = KilatDataCollator(
        pad_token_id=model_cfg.pad_token_id,
        max_length=dl_cfg.max_seq_length,
    )

    if dl_cfg.dataset_format == "synthetic":
        train_ds: Dataset = SyntheticTokenDataset(
            vocab_size=model_cfg.vocab_size,
            seq_len=dl_cfg.max_seq_length,
            num_samples=dl_cfg.synthetic_num_samples,
            pad_token_id=model_cfg.pad_token_id,
            seed=train_cfg.seed,
        )
        eval_ds: Optional[Dataset] = SyntheticTokenDataset(
            vocab_size=model_cfg.vocab_size,
            seq_len=dl_cfg.max_seq_length,
            num_samples=dl_cfg.synthetic_eval_samples,
            pad_token_id=model_cfg.pad_token_id,
            seed=train_cfg.seed + 10_007,  # Disjoint seed guarantees zero train/val leakage
        )
    else:
        if not dl_cfg.train_data_path:
            raise ValueError(
                "dataloader.train_data_path must be specified when dataset_format != 'synthetic'."
            )
        train_ds = PretrainingDataset(
            source=dl_cfg.train_data_path,
            chunk_size=dl_cfg.max_seq_length,
            memmap_stride=dl_cfg.memmap_stride,
            return_tensors=True,
        )
        if dl_cfg.use_packing:
            train_ds = PackedDataset(
                train_ds,
                block_size=dl_cfg.packed_block_size or dl_cfg.max_seq_length,
                eos_token_id=model_cfg.eos_token_id,
                pad_token_id=model_cfg.pad_token_id,
                drop_last=dl_cfg.drop_last,
            )

        if dl_cfg.eval_data_path:
            eval_ds = PretrainingDataset(
                source=dl_cfg.eval_data_path,
                chunk_size=dl_cfg.max_seq_length,
                memmap_stride=dl_cfg.memmap_stride,
                return_tensors=True,
            )
        else:
            eval_ds = None

    train_loader = build_train_dataloader(
        dataset=train_ds,
        batch_size=dl_cfg.train_batch_size,
        collate_fn=collator,
        num_workers=dl_cfg.num_workers,
        pin_memory=dl_cfg.pin_memory,
        prefetch_factor=dl_cfg.prefetch_factor,
        drop_last=dl_cfg.drop_last,
        persistent_workers=dl_cfg.persistent_workers,
        seed=train_cfg.seed,
        use_distributed_sampler=dl_cfg.use_distributed_sampler,
    )

    eval_loader: Optional[DataLoader] = None
    if eval_ds is not None:
        eval_loader = build_eval_dataloader(
            dataset=eval_ds,
            batch_size=dl_cfg.eval_batch_size,
            collate_fn=collator,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
            prefetch_factor=dl_cfg.prefetch_factor,
            drop_last=False,
            persistent_workers=dl_cfg.persistent_workers,
            seed=train_cfg.seed,
            use_distributed_sampler=dl_cfg.use_distributed_sampler,
        )

    return train_loader, eval_loader
