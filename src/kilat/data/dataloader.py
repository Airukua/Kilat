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
    distributed_shuffle: bool = True,
    distributed_seed: Optional[int] = None,
    **kwargs,
) -> DataLoader:
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

    # Worker init function for reproducibility across processes.
    def _worker_init_fn(wid: int) -> None:
        worker_init_fn(wid, base_seed=seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
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


# ---------------------------------------------------------------------------
# Convenience wrappers for training and evaluation
# ---------------------------------------------------------------------------

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
    use_distributed_sampler: bool = True,
    **kwargs,
) -> DataLoader:
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
    sampler = dataloader.sampler
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    # Some DataLoaders use a batch_sampler instead (e.g., when using
    # a custom batch sampler). Check and set epoch there too.
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)