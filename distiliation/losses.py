from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class DistillLossOutput:
    """Container for distillation loss components – all values are logged."""
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    ce_loss: torch.Tensor
    alpha: float                      # weight for soft labels
    temperature: float

    def log_dict(self) -> dict[str, float]:
        return {
            "loss/total":       self.total_loss.item(),
            "loss/kl":          self.kl_loss.item(),
            "loss/ce":          self.ce_loss.item(),
            "distill/alpha":    self.alpha,
            "distill/temperature": self.temperature,
        }


class BaseDistillLoss(ABC, nn.Module):
    """
    Interface for all distillation loss functions.

    WHY: Different distillation strategies (vanilla KD, reverse KL, adaptive temp)
    share the same signature. This abstraction allows the trainer to switch
    losses without changing the training loop.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        student_logits: torch.Tensor,   # [B, T, V]
        teacher_logits: torch.Tensor,   # [B, T, V]
        labels: torch.Tensor,           # [B, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> DistillLossOutput:
        ...


class VanillaKDLoss(BaseDistillLoss):
    """
    Classical distillation loss from Hinton et al., 2015.

        L = α · T² · KL(student || teacher) + (1 - α) · CE(student, labels)

    WHY T² scaling: The gradient magnitude scales with 1/T² when using softmax
    with temperature. Multiplying by T² keeps the contribution of soft targets
    comparable to hard targets regardless of temperature.

    Args:
        temperature: Softmax temperature to soften teacher distributions.
                     Higher T → softer probabilities, more information per token.
        alpha: Weight between soft labels (KL) and hard labels (CE).
               0.0 = pure CE, 1.0 = pure KD.
        ignore_index: Token index to ignore in CE loss (e.g., padding).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        # Validation: alpha and temperature must be in valid ranges.
        assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"
        assert temperature > 0, "temperature must be positive"

        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DistillLossOutput:
        T = self.temperature
        # Apply temperature scaling before log/softmax.
        soft_student = F.log_softmax(student_logits / T, dim=-1)   # [B, T, V]
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)       # [B, T, V]

        # KL divergence per token, summed over vocab dimension.
        kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(-1)  # [B, T]

        if attention_mask is not None:
            # Mask out padding tokens. Use sum over mask for mean.
            kl = (kl * attention_mask).sum() / attention_mask.sum()
        else:
            kl = kl.mean()

        # Rescale gradient as per original KD paper.
        kl_loss = kl * (T ** 2)

        B, Tlen, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.view(B * Tlen, V),
            labels.view(B * Tlen),
            ignore_index=self.ignore_index,
        )

        total = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return DistillLossOutput(
            total_loss=total,
            kl_loss=kl_loss,
            ce_loss=ce_loss,
            alpha=self.alpha,
            temperature=self.temperature,
        )


class ReverseKDLoss(BaseDistillLoss):
    """
    Reverse KL: KL(teacher || student) – used in MiniLLM.

    WHY reverse KL instead of forward KL?
    - Forward KL (KL(student||teacher)) encourages the student to cover all
      modes of the teacher, even improbable ones → can cause hallucinations.
    - Reverse KL (KL(teacher||student)) forces the student to focus on the
      teacher's most confident modes (mode‑seeking). This reduces hallucinations
      and yields better calibration for generative tasks.

    Trade‑off: Reverse KL can be less stable and may underfit diverse outputs.
    Usually paired with a small CE term (α ≈ 0.5) to retain hard label signal.

    Args:
        temperature: Softmax temperature. Reverse KL often works well with T=1.0
                     because mode‑seeking is already aggressive.
        alpha: Weight for KL vs CE. Higher α gives more mode‑seeking behavior.
        ignore_index: Token index to ignore in CE loss.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DistillLossOutput:
        T = self.temperature
        # Reverse KL: KL(teacher || student) = Σ p_t · log(p_t / p_s)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)         # [B, T, V]
        log_student  = F.log_softmax(student_logits / T, dim=-1)     # [B, T, V]
        log_teacher  = F.log_softmax(teacher_logits / T, dim=-1)     # [B, T, V]

        # Per‑token KL: Σ teacher * (log_teacher - log_student)
        kl = (soft_teacher * (log_teacher - log_student)).sum(-1)    # [B, T]

        if attention_mask is not None:
            kl = (kl * attention_mask).sum() / attention_mask.sum()
        else:
            kl = kl.mean()

        # Also rescale by T² to keep magnitude independent of temperature.
        kl_loss = kl * (T ** 2)

        B, Tlen, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.view(B * Tlen, V),
            labels.view(B * Tlen),
            ignore_index=self.ignore_index,
        )

        total = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return DistillLossOutput(
            total_loss=total,
            kl_loss=kl_loss,
            ce_loss=ce_loss,
            alpha=self.alpha,
            temperature=self.temperature,
        )


class AdaptiveKDLoss(BaseDistillLoss):
    """
    KL loss with a learnable temperature parameter.

    WHY: The optimal temperature may change during training or vary across
    datasets. Hardcoding T is suboptimal. By making temperature learnable,
    the model can automatically adjust the softness of the teacher distribution.

    HOW: We parameterize T in log‑space to keep it positive, then clamp to a
    safe range [temp_min, temp_max] to avoid extreme values that could cause
    gradient explosion or vanishing.

    Args:
        init_temperature: Initial temperature value (default 4.0).
        alpha: Weight for KL vs CE.
        temp_min: Minimum allowed temperature (prevents T < 1, which makes
                  distributions sharper and may hinder distillation).
        temp_max: Maximum allowed temperature (prevents overly flat distributions
                  that lose signal).
        ignore_index: Token index to ignore in CE loss.
    """

    def __init__(
        self,
        init_temperature: float = 4.0,
        alpha: float = 0.5,
        temp_min: float = 1.0,
        temp_max: float = 10.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.ignore_index = ignore_index

        # Store log(T) so that exponentiation keeps T > 0 automatically.
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    @property
    def temperature(self) -> float:
        T = self.log_temperature.exp().clamp(self.temp_min, self.temp_max)
        return T.item()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DistillLossOutput:
        T = self.log_temperature.exp().clamp(self.temp_min, self.temp_max)

        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)

        kl = F.kl_div(soft_student, soft_teacher, reduction="none").sum(-1)

        if attention_mask is not None:
            kl = (kl * attention_mask).sum() / attention_mask.sum()
        else:
            kl = kl.mean()

        kl_loss = kl * (T ** 2)

        B, Tlen, V = student_logits.shape
        ce_loss = F.cross_entropy(
            student_logits.view(B * Tlen, V),
            labels.view(B * Tlen),
            ignore_index=self.ignore_index,
        )

        total = self.alpha * kl_loss + (1 - self.alpha) * ce_loss

        return DistillLossOutput(
            total_loss=total,
            kl_loss=kl_loss,
            ce_loss=ce_loss,
            alpha=self.alpha,
            temperature=T.item(),
        )

    def log_dict(self) -> dict[str, float]:
        """Extra logging to track the learned temperature during training."""
        return {"distill/learned_temperature": self.temperature}


_REGISTRY: dict[str, type[BaseDistillLoss]] = {
    "vanilla":   VanillaKDLoss,
    "reverse":   ReverseKDLoss,
    "adaptive":  AdaptiveKDLoss,
}

def build_loss(name: str, **kwargs) -> BaseDistillLoss:
    """
    Factory function to construct a distillation loss by name.

    WHY: Decouples loss creation from the training script. The trainer only
    needs the loss name and config, not the concrete class.

    Example:
        loss_fn = build_loss("vanilla", temperature=4.0, alpha=0.5)
        loss_fn = build_loss("reverse", temperature=1.0, alpha=0.7)
        loss_fn = build_loss("adaptive", init_temperature=4.0, alpha=0.5)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)

def register_loss(name: str, cls: type[BaseDistillLoss]):
    """Register a custom loss class for use with build_loss()."""
    if not issubclass(cls, BaseDistillLoss):
        raise TypeError(f"{cls} must be a subclass of BaseDistillLoss")
    _REGISTRY[name] = cls