"""Canonical neural architecture primitives for Kilat."""

from kilat.models.base import BaseConfig, BasePreTrainedModel
from kilat.models.attention import KilatAttention, KilatAttentionRoPE, LayerHybridCache
from kilat.models.decay import GlobalDecayLayer, run_global_decay_recurrence
from kilat.models.mla import LatentAttentionLayer, apply_rope_to_tokens
from kilat.models.blocks import Block, RMSNorm
from kilat.models.ffn import DeepSeekMoE, FeedForward, SwiGLU
from kilat.models.kernels.flash_decay import (
    HAS_TRITON,
    FlashDecayFunction,
    compute_decay_normalizer_from_logit,
    flash_decay_fwd_kernel,
    triton_global_decay,
)
from kilat.models.rope import (
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    build_rope_cache,
    rope,
)
from kilat.models.transformer import (
    KilatModelOutput,
    KilatPreTrainedModel,
    KilatTransformer,
)

__all__ = [
    "BaseConfig",
    "BasePreTrainedModel",
    "KilatAttention",
    "KilatAttentionRoPE",
    "GlobalDecayLayer",
    "run_global_decay_recurrence",
    "LatentAttentionLayer",
    "apply_rope_to_tokens",
    "LayerHybridCache",
    "RMSNorm",
    "Block",
    "SwiGLU",
    "DeepSeekMoE",
    "FeedForward",
    "KilatPreTrainedModel",
    "KilatTransformer",
    "KilatModelOutput",
    "apply_rotary_pos_emb",
    "build_rope_cache",
    "RotaryPositionalEmbedding",
    "rope",
    "HAS_TRITON",
    "FlashDecayFunction",
    "flash_decay_fwd_kernel",
    "triton_global_decay",
    "compute_decay_normalizer_from_logit",
]
