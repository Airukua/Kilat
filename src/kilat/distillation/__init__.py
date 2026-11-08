"""Canonical knowledge distillation losses, student, and teacher interfaces."""

from kilat.distillation.losses import (
    AdaptiveKDLoss,
    BaseDistillLoss,
    DistillLossOutput,
    ReverseKDLoss,
    VanillaKDLoss,
    build_loss,
    register_loss,
)
from kilat.distillation.student import BaseStudent
from kilat.distillation.teacher import BaseTeacher

__all__ = [
    "DistillLossOutput",
    "BaseDistillLoss",
    "VanillaKDLoss",
    "ReverseKDLoss",
    "AdaptiveKDLoss",
    "build_loss",
    "register_loss",
    "BaseStudent",
    "BaseTeacher",
]
