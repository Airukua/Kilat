from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseStudent(ABC, nn.Module):
    """
    Interface for all student models in distillation.

    WHY: Student model is trainable (unlike teacher). It can be initialized from
    scratch or from a previous checkpoint. The abstraction allows swapping
    between different architectures (Kilat, HuggingFace, custom) without
    changing the distillation loop.

    Key design:
    - `load()`: resume from a checkpoint (e.g., after interruption).
    - `from_config()`: create a new untrained student.
    - `forward()` must return logits matching teacher's output shape.
    - No freezing by default – student is trainable.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> "BaseStudent":
        """Load student from an existing checkpoint (resume training)."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, *args, **kwargs) -> "BaseStudent":
        """Create a new student from a configuration (train from scratch)."""
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return logits: Tensor [batch, seq_len, vocab_size]."""
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size; must match teacher's vocab for KL divergence."""
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        try:
            n = self.num_parameters / 1e6
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
            return (
                f"{self.__class__.__name__}"
                f"({n:.1f}M params, {trainable:.1f}M trainable)"
            )
        except Exception:
            return f"{self.__class__.__name__}(trainable)"


