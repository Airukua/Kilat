from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class BaseTeacher(ABC, nn.Module):
    """
    Interface for all teacher models in distillation.

    WHY: Distillation requires a frozen teacher that provides logits/soft labels.
    This abstraction allows swapping between different teacher sources (local
    Kilat checkpoints, HuggingFace models, custom APIs) without changing student code.

    Key design decisions:
    - Teacher is always frozen (no gradients, no training). We enforce this by
      overriding `train()` to be a no‑op and calling `_freeze()` in subclasses.
    - `forward()` must return logits of shape [B, T, vocab_size] (not probabilities).
    - `load()` is a factory method per subclass → each backend can have different
      loading arguments (e.g., device, dtype, trust_remote_code).
    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> "BaseTeacher":
        """Load teacher from any source (checkpoint, HF hub, API, etc.)."""
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

    def _freeze(self):
        """Freeze all parameters and set to eval mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """
        Override to prevent accidental training. Teacher stays in eval mode always.

        WHY: Some training loops call `model.train()` on all modules. This override
        ensures teacher never leaves eval mode, which would break distillation
        (e.g., dropout would introduce noise, batch norm would track statistics).
        """
        return self

    @property
    def vocab_size(self) -> int:
        """Return teacher's vocabulary size (must match student's for distillation loss)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            n = sum(p.numel() for p in self.parameters()) / 1e6
            return f"{self.__class__.__name__}({n:.1f}M params, frozen)"
        except Exception:
            return f"{self.__class__.__name__}(frozen)"


