"""
Kilat Hugging Face Integration Bridge.

Provides explicit, opt-in registration and interoperability with the Hugging Face
transformers ecosystem (AutoConfig, AutoModel, AutoModelForCausalLM).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def register_kilat() -> None:
    """
    Explicitly register Kilat architecture with Hugging Face transformers.

    Registers:
        - AutoConfig: 'kilat' -> KilatConfig
        - AutoModel: KilatConfig -> KilatModel
        - AutoModelForCausalLM: KilatConfig -> KilatForCausalLM

    Safe to call multiple times (idempotent). Does not raise if transformers
    is not installed.
    """
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        from .configuration_kilat import KilatConfig
        from .modeling_kilat import KilatForCausalLM, KilatModel

        try:
            AutoConfig.register("kilat", KilatConfig)
        except Exception as exc:
            logger.debug("AutoConfig registration skipped: %s", exc)

        try:
            AutoModel.register(KilatConfig, KilatModel)
        except Exception as exc:
            logger.debug("AutoModel registration skipped: %s", exc)

        try:
            AutoModelForCausalLM.register(KilatConfig, KilatForCausalLM)
        except Exception as exc:
            logger.debug("AutoModelForCausalLM registration skipped: %s", exc)

        logger.debug("Successfully registered Kilat with Hugging Face AutoClasses.")
    except ImportError:
        logger.debug("transformers is not installed; skipping AutoClass registration.")


__all__ = ["register_kilat"]
