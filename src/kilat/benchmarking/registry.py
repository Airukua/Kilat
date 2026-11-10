"""Model architecture registry and unified adapter for Kilat certification.

Allows plug-and-play evaluation of Kilat configurations and baseline models.
Transparently handles both KilatModelOutput (CausalLMOutputWithPast) and raw Tensors.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from kilat.configs.model_config import KilatConfig
from kilat.models.transformer import KilatTransformer

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def extract_logits(output: Any) -> torch.Tensor:
    """Safely extract logits tensor from model output.

    Handles:
        - KilatModelOutput / CausalLMOutputWithPast (output.logits)
        - Dict with 'logits' key
        - Raw Tensor output
    """
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Cannot extract logits from output type {type(output)}")


class ModelWrapper(nn.Module):
    """Wrapper that ensures model(input_ids) returns a clean logits Tensor."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        out = self.inner(input_ids, **kwargs)
        return extract_logits(out)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def register_model(name: str):
    """Decorator to register a model builder function."""
    def _wrap(builder_fn: Callable[..., nn.Module]):
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered in Kilat benchmark registry.")
        _REGISTRY[name] = builder_fn
        return builder_fn
    return _wrap


def build_model(name: str, wrap_for_logits: bool = True, **kwargs: Any) -> nn.Module:
    """Build a registered model by name.

    Args:
        name: Name of registered model.
        wrap_for_logits: If True, wraps model with ModelWrapper so forward() returns raw logits tensor.
        **kwargs: Overrides for model hyperparameters.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(empty)"
        raise KeyError(
            f"Model '{name}' not found in benchmark registry. Available models: {available}."
        )
    model = _REGISTRY[name](**kwargs)
    if wrap_for_logits and not isinstance(model, ModelWrapper):
        return ModelWrapper(model)
    return model


def list_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_REGISTRY.keys())


# =====================================================================
# Built-in Reference Architectures & Kilat Presets
# =====================================================================

class TinyCausalAttention(nn.Module):
    """Reference vanilla transformer with standard quadratic causal attention."""

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos_ids)
        # In PyTorch >= 2.0, generate_square_subsequent_mask produces float mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        x = self.layers(x, mask=causal_mask, is_causal=True)
        return self.lm_head(x)


@register_model("reference-attention")
def _build_reference(
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    **_: Any,
) -> nn.Module:
    return TinyCausalAttention(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
    )


def _build_kilat_preset(
    recall_ratio: float,
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    recall_pattern: str = "uniform",
    **kwargs: Any,
) -> KilatTransformer:
    head_dim = d_model // n_heads
    cfg = KilatConfig(
        vocab_size=vocab_size,
        n_embd=d_model,
        n_layer=n_layers,
        n_head=n_heads,
        recall_ratio=recall_ratio,
        recall_pattern=recall_pattern,
        latent_dim=kwargs.pop("latent_dim", max(32, d_model)),
        q_lora_rank=kwargs.pop("q_lora_rank", None),
        use_fusion_gate=kwargs.pop("use_fusion_gate", False),
        use_rope=True,
        rope_head_dim=kwargs.pop("rope_head_dim", max(8, head_dim if head_dim % 2 == 0 else head_dim - 1)),
        ffn_mode="swiglu",
        max_position_embeddings=max_seq_len,
        tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
        initializer_range=kwargs.pop("initializer_range", None),
        **kwargs,
    )
    return KilatTransformer(cfg)


@register_model("kilat-v2")
@register_model("kilat-hybrid")
def _build_kilat_hybrid(
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    **kwargs: Any,
) -> KilatTransformer:
    """Kilat v2 Hybrid Sub-Quadratic (Global Decay + Latent MLA at recall_ratio=0.5 with Decoupled Synergy)."""
    return _build_kilat_preset(
        recall_ratio=0.5,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        **kwargs,
    )


@register_model("kilat-macro-synergy")
def _build_kilat_macro_synergy(
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    **kwargs: Any,
) -> KilatTransformer:
    """Kilat Macro Synergy (Interleaved Stacking: L0 Global Decay -> L1 Latent MLA)."""
    return _build_kilat_preset(
        recall_ratio=0.5,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        recall_pattern="macro_synergy",
        **kwargs,
    )


@register_model("kilat-pure-decay")
def _build_kilat_decay(
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    **kwargs: Any,
) -> KilatTransformer:
    """Kilat Pure Sub-Quadratic Global Decay (recall_ratio=0.0)."""
    return _build_kilat_preset(
        recall_ratio=0.0,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        **kwargs,
    )


@register_model("kilat-pure-attention")
def _build_kilat_attention(
    vocab_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    max_seq_len: int = 4096,
    **kwargs: Any,
) -> KilatTransformer:
    """Kilat Pure Latent MLA Attention (recall_ratio=1.0)."""
    return _build_kilat_preset(
        recall_ratio=1.0,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        **kwargs,
    )
