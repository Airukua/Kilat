"""
Configuration class for Kilat architecture on Hugging Face Hub.

Standalone and self-contained; depends ONLY on `transformers`.
"""

from __future__ import annotations
from typing import Optional, List, Union, Any
from transformers import PretrainedConfig


class KilatConfig(PretrainedConfig):
    """
    Configuration class for KilatTransformer on the Hugging Face Hub ecosystem.
    Compatible with `AutoConfig` via `trust_remote_code=True` and local registration.
    """

    model_type = "kilat"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Core dimensions & architecture (max <= 13 parameters to satisfy SonarQube S107)
        vocab_size: int = 32000,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        recall_ratio: float = 0.5,
        recall_pattern: str = "macro_synergy",
        use_rope: bool = False,
        ffn_mode: str = "moe",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs: Any,
    ):
        # Resolve aliases if provided in kwargs
        if "d_model" in kwargs and n_embd == 768:
            n_embd = kwargs.pop("d_model")
        if "n_heads" in kwargs and n_head == 12:
            n_head = kwargs.pop("n_heads")
        if "n_layers" in kwargs and n_layer == 12:
            n_layer = kwargs.pop("n_layers")
        if "max_seq_len" in kwargs and "max_position_embeddings" not in kwargs:
            kwargs["max_position_embeddings"] = kwargs.pop("max_seq_len")

        # Extract attention & RoPE parameters
        recall_ratios: Optional[List[float]] = kwargs.pop("recall_ratios", None)
        latent_dim: Optional[int] = kwargs.pop("latent_dim", None)
        q_lora_rank: Optional[int] = kwargs.pop("q_lora_rank", None)
        use_fusion_gate: bool = kwargs.pop("use_fusion_gate", False)
        attn_drop: float = kwargs.pop("attn_drop", 0.0)
        rope_head_dim: Optional[int] = kwargs.pop("rope_head_dim", None)
        rope_base: float = kwargs.pop("rope_base", 10000.0)
        yarn_scale: Optional[float] = kwargs.pop("yarn_scale", None)
        yarn_base_seq_len: int = kwargs.pop("yarn_base_seq_len", 4096)

        # Extract FFN & MoE parameters
        ff_mult: float = kwargs.pop("ff_mult", 8 / 3)
        ffn_dropout: float = kwargs.pop("ffn_dropout", 0.0)
        num_experts: int = kwargs.pop("num_experts", 8)
        active_experts: int = kwargs.pop("active_experts", 2)
        num_shared_experts: int = kwargs.pop("num_shared_experts", 0)
        fine_grained_factor: int = kwargs.pop("fine_grained_factor", 1)
        aux_loss_coef: float = kwargs.pop("aux_loss_coef", 0.01)
        device_balance_coef: float = kwargs.pop("device_balance_coef", 0.0)
        router_balance_mode: str = kwargs.pop("router_balance_mode", "aux_loss")

        # Extract Regularization & Stability parameters
        embd_drop: float = kwargs.pop("embd_drop", 0.0)
        resid_drop: float = kwargs.pop("resid_drop", 0.0)
        z_loss_coef: float = kwargs.pop("z_loss_coef", 0.0)
        debug_nan_checks: bool = kwargs.pop("debug_nan_checks", False)
        initializer_range: float = kwargs.pop("initializer_range", 0.02)
        init_scheme: str = kwargs.pop("init_scheme", "gpt2")
        tie_word_embeddings: bool = kwargs.pop("tie_word_embeddings", True)
        use_cache: bool = kwargs.pop("use_cache", False)

        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.recall_ratio = recall_ratio
        self.recall_pattern = recall_pattern
        self.recall_ratios = recall_ratios
        self.latent_dim = latent_dim if latent_dim is not None else (n_embd // 4)
        self.q_lora_rank = q_lora_rank
        self.use_fusion_gate = use_fusion_gate
        self.attn_drop = attn_drop

        self.use_rope = use_rope
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else (self.head_dim // 2)
        self.rope_base = rope_base
        self.yarn_scale = yarn_scale
        self.yarn_base_seq_len = yarn_base_seq_len

        self.ffn_mode = ffn_mode
        self.ff_mult = ff_mult
        self.ffn_dropout = ffn_dropout

        self.num_experts = num_experts
        self.active_experts = active_experts
        self.num_shared_experts = num_shared_experts
        self.fine_grained_factor = fine_grained_factor
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef
        self.router_balance_mode = router_balance_mode

        self.embd_drop = embd_drop
        self.resid_drop = resid_drop
        self.z_loss_coef = z_loss_coef
        self.debug_nan_checks = debug_nan_checks
        self.initializer_range = initializer_range
        self.init_scheme = init_scheme
        self.use_cache = use_cache
