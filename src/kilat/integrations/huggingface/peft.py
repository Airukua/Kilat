"""
Kilat Parameter-Efficient Fine-Tuning (PEFT) LoRA Configuration & Helpers.

Provides canonical LoRA adaptation configurations tailored specifically for Kilat's
hybrid architecture (Macro-Synergy: Global Exponential Decay + Latent MLA).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Canonical linear projection targets in Kilat:
# - SwiGLU / MoE FFN: gate, up, down
# - Latent MLA: q_proj, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj_recall
# - Global Decay: v_proj_global, o_proj_global
DEFAULT_KILAT_LORA_TARGET_MODULES: list[str] = [
    "gate",
    "up",
    "down",
    "q_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj",
    "kv_b_proj",
    "o_proj_recall",
    "v_proj_global",
    "o_proj_global",
]

# Recurrent dynamics parameters that must NEVER be modified by low-rank linear adapters
FORBIDDEN_LORA_TARGETS: frozenset[str] = frozenset(
    {"log_lambda", "lambda", "decay_lambda", "decay_bias"}
)


def get_kilat_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[Sequence[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    **kwargs: Any,
) -> Any:
    """
    Build a validated Hugging Face PEFT LoraConfig for Kilat models.

    Parameters
    ----------
    r : int
        LoRA attention rank dimension (default 16).
    lora_alpha : int
        LoRA alpha scaling factor (default 32).
    target_modules : Optional[Sequence[str]]
        Target linear modules. Defaults to all Kilat linear projections
        (FFN SwiGLU, Latent MLA, and Global Decay projections).
    lora_dropout : float
        Dropout probability for LoRA layers (default 0.05).
    bias : str
        Bias type for LoRA ('none', 'all', 'lora_only'). Default 'none'.
    task_type : str
        Task type, default 'CAUSAL_LM'.
    **kwargs : Any
        Additional keyword arguments forwarded to peft.LoraConfig.

    Returns
    -------
    LoraConfig
        Configured and validated PEFT LoraConfig instance.

    Raises
    ------
    ImportError
        If `peft` is not installed.
    ValueError
        If `target_modules` attempts to adapt forbidden recurrent parameters.
    """
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "The 'peft' package is required for parameter-efficient fine-tuning. "
            "Install it via: pip install peft"
        ) from exc

    resolved_targets = list(target_modules) if target_modules is not None else list(DEFAULT_KILAT_LORA_TARGET_MODULES)

    # Invariant guard: Prevent corrupting recurrent decay dynamics with linear adapters
    forbidden_present = [m for m in resolved_targets if any(f in m.lower() for f in FORBIDDEN_LORA_TARGETS)]
    if forbidden_present:
        raise ValueError(
            f"Cannot apply LoRA to recurrent dynamics parameters: {forbidden_present}. "
            "Kilat's decay rates (log_lambda) require analytical recurrence stability "
            "and must remain frozen or tuned via native full-rank optimizer."
        )

    # Map string task_type to TaskType enum if applicable
    resolved_task_type = getattr(TaskType, task_type, task_type)

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=resolved_targets,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=resolved_task_type,
        **kwargs,
    )
