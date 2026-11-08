"""
Single-source-of-truth distributed execution helpers (`DistributedContext`).

Consolidates rank, world size, barrier, and collective reduction queries across
training, evaluation, checkpointing, and data loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def is_distributed_initialized() -> bool:
    """Return True if ``torch.distributed`` is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def is_distributed() -> bool:
    """Alias of ``is_distributed_initialized`` for dataloader compatibility."""
    return is_distributed_initialized()


def get_global_rank() -> int:
    """Return the global rank of the current process, or ``0`` if non-distributed."""
    return dist.get_rank() if is_distributed_initialized() else 0


def get_local_rank() -> int:
    """Return local rank from ``LOCAL_RANK`` env var when distributed, else ``0``."""
    if not is_distributed_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return distributed world size, or ``1`` if non-distributed."""
    return dist.get_world_size() if is_distributed_initialized() else 1


def is_main_process() -> bool:
    """Return True on global rank 0 (or in single-process execution)."""
    return get_global_rank() == 0


def barrier() -> None:
    """Synchronize all ranks if ``torch.distributed`` is initialized."""
    if is_distributed_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place ``SUM`` all-reduce across ranks when distributed; no-op otherwise."""
    if is_distributed_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@dataclass(frozen=True)
class DistributedContext:
    """Immutable snapshot of the active distributed process group topology."""

    initialized: bool
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def current(cls) -> "DistributedContext":
        """Capture the live process-group state."""
        return cls(
            initialized=is_distributed_initialized(),
            rank=get_global_rank(),
            local_rank=get_local_rank(),
            world_size=get_world_size(),
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        barrier()

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return all_reduce_sum(tensor)


def setup_distributed(backend: str | None = None) -> DistributedContext:
    """
    Initialize `torch.distributed` process group if `WORLD_SIZE` environment variable is > 1.

    If already initialized, returns `DistributedContext.current()` immediately.
    Default backend: 'nccl' if `torch.cuda.is_available()` else 'gloo'.
    """
    if is_distributed_initialized():
        return DistributedContext.current()

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None or int(world_size_env) <= 1:
        return DistributedContext.current()

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend)
    return DistributedContext.current()


def cleanup_distributed() -> None:
    """Destroy `torch.distributed` process group if currently initialized."""
    if is_distributed_initialized():
        dist.destroy_process_group()


__all__ = [
    "DistributedContext",
    "is_distributed_initialized",
    "is_distributed",
    "get_global_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "all_reduce_sum",
    "setup_distributed",
    "cleanup_distributed",
]
