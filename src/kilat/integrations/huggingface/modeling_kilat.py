"""
Kilat Hugging Face Adapter: Thin, Zero-Duplication Ecosystem Bridge.

Adheres to Senior Software Architecture Standard (Pillar 7 & 8):
- Single Source of Truth (SSOT): Composes canonical `kilat.models.transformer.KilatTransformer`
- Zero duplicated RMSNorm, SwiGLU, MoE routing, or Triton attention kernels
- File size strictly < 100 lines (adheres to < 400 lines hard ceiling)
"""

from __future__ import annotations

from typing import Union

from kilat.configs.model_config import KilatConfig as KilatNativeConfig
from kilat.models.transformer import KilatTransformer
from .configuration_kilat import KilatConfig


class KilatModel(KilatTransformer):
    """
    Base Kilat Transformer module for Hugging Face AutoModel compatibility.

    Directly subclasses canonical KilatTransformer, providing 100% architectural
    and numerical equivalence without code duplication.
    """

    config_class = KilatConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Union[KilatConfig, KilatNativeConfig]):
        if isinstance(config, KilatConfig):
            native_cfg = KilatNativeConfig(**config.to_dict())
            super().__init__(native_cfg)
            self.config = config
        else:
            super().__init__(config)


class KilatForCausalLM(KilatTransformer):
    """
    Causal Language Modeling adapter for Hugging Face AutoModelForCausalLM.

    Provides complete compatibility with Hugging Face GenerationMixin, PEFT LoRA,
    TRL, and external evaluation harnesses while delegating core computation directly
    to Kilat's canonical engine.
    """

    config_class = KilatConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Union[KilatConfig, KilatNativeConfig]):
        if isinstance(config, KilatConfig):
            native_cfg = KilatNativeConfig(**config.to_dict())
            super().__init__(native_cfg)
            self.config = config
        else:
            super().__init__(config)
