from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from .blocks import Block, RMSNorm
from kilat.configs.model_config import KilatConfig
from kilat.configs.main_config import MainConfig
from kilat.pipeline.generation.generation_mixin import GenerationMixin
from kilat.utils.base_model import (
    BasePreTrainedModel,
    CONFIG_NAME,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX,
    WEIGHTS_INDEX_NAME,
    _SAFETENSORS_AVAILABLE,
    _st_load,
)
from kilat.utils.validators import (
    validate_finite_tensor,
    validate_tensor_rank,
)

logger = logging.getLogger(__name__)


class KilatPreTrainedModel(BasePreTrainedModel):
    """
    Abstract base for KilatTransformer – handles weight init and HF integration.

    This class bridges the gap between a custom transformer architecture and
    the HuggingFace ecosystem. By subclassing ``PreTrainedModel``, we get:

    - **Serialization**: ``from_pretrained`` / ``save_pretrained`` work out of
      the box, including sharded checkpoints and safetensors.
    - **Gradient checkpointing**: ``supports_gradient_checkpointing = True``
      enables activation recomputation for memory‑efficient training.
    - **Weight initialization**: ``_init_weights`` is called automatically by
      ``post_init()`` during construction and after loading checkpoints.
    - **Config integration**: ``config_class`` tells HF which config class to
      use when loading config.json alongside model weights.

    Weight Initialization Strategy
    -----------------------------
    N(0, 0.02) for weights, zero for biases. This follows GPT‑2 initialization
    and is standard for decoder‑only transformers trained with AdamW:
    - σ = 0.02 is a "small" initialization that prevents early saturation of
      softmax/activation functions while allowing gradients to flow.
    - Biases are zero‑initialized because the normalization layers (RMSNorm)
      handle distribution shifting — no need for non‑zero bias priors.
    - For residual networks, small initialization prevents the residual stream
      from being dominated by early layers' random outputs.

    Subclass this to inherit standard Hugging Face functionality:
        - ``from_pretrained`` / ``save_pretrained``
        - Gradient checkpointing support
        - Shared weight‑initialisation logic
    """
    config_class = KilatConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        """
        Default initialisation: Linear/Embedding weights ~ N(0, 0.02), biases = 0.

        Called automatically by HuggingFace's ``post_init()`` for every submodule
        during model construction. Also called when loading weights from a
        checkpoint for modules that are NOT found in the checkpoint (new layers
        added after pretraining, e.g., fine‑tuning heads).

        The use of ``torch.nn.init.normal_`` (in‑place) rather than creating
        new tensors preserves any existing parameter attributes (e.g., device,
        dtype) and works correctly with parameter sharding.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class KilatTransformer(KilatPreTrainedModel, GenerationMixin):
    """
    Hugging Face‑compatible KilatTransformer for causal language modelling with generation support.

    Architecture Overview
    --------------------
    This is a standard decoder‑only transformer (GPT‑style) with the
    KilatTransformer enhancements:

    1. **Embedding layer**: Maps token IDs to continuous vectors.
    2. **Dropout**: Applied to embeddings (optional, configurable).
    3. **Transformer blocks**: Stack of ``Block`` modules, each containing:
       - KilatAttention (global decay + latent MLA) or KilatAttentionRoPE (with RoPE)
       - FeedForward (dense SwiGLU or DeepSeek‑V2 MoE)
    4. **Final RMSNorm**: Pre‑output normalization (standard in pre‑norm arch).
    5. **LM Head**: Linear projection to vocabulary size for next‑token prediction.

    Weight Tying
    -----------
    The input embedding matrix (``wte``) and output projection (``lm_head``)
    share the same weight tensor. This is standard in modern LLMs because:
    - Saves ``vocab_size * n_embd`` parameters (typically 30‑40% of total for
      small/medium models, decreasing as depth increases)
    - Empirically performs as well as separate weights (Press & Wolf, 2017)
    - The embedding matrix already learns good token representations; reusing
      it for output projection means the model "reads" and "writes" in the
      same space

    RoPE Support (NEW)
    -----------------
    The model now supports Decoupled RoPE via the `use_rope` config flag.
    When enabled, KilatAttentionRoPE is used instead of KilatAttention:
    - Adds positional awareness to MLA recall path
    - Cache structure becomes (global_state, latent_kv, rope_k)
    - Enables better long-range position handling

    MoE Auxiliary Loss Handling
    ---------------------------
    When any block uses MoE (``ffn_mode='moe'``), the forward pass accumulates
    auxiliary load‑balancing losses from all blocks. This loss is added to the
    primary cross‑entropy loss so it backpropagates through all parameters.
    The individual layer losses are already multiplied by ``aux_loss_coef``
    in their respective blocks; the sum here simply aggregates them.

    HuggingFace Integration Details
    -------------------------------
    - **kwargs absorption**: The ``**kwargs`` parameter absorbs extra arguments
      that HuggingFace's Trainer may pass (e.g., ``attention_mask``, ``token_type_ids``)
      even when the model doesn't use them. Without this, Trainer raises
      TypeError due to unexpected keyword arguments.
    - **return_dict**: Respects both the function argument and the config default,
      enabling tuple output for backward compatibility with older HF code.
    - **CausalLMOutputWithPast**: Includes ``past_key_values=None`` as a
      placeholder. Full KV‑cache support requires implementing a cache object
      that tracks the compressed MLA KV states.

    Generation Support (via GenerationMixin)
    -----------------------------------------
    The model now inherits from ``GenerationMixin``, which provides the ``generate()``
    method for text generation. This supports:
    - Greedy decoding (do_sample=False, num_beams=1)
    - Multinomial sampling (do_sample=True, num_beams=1)
    - Beam search (num_beams>1, do_sample=False)
    - Various sampling strategies (temperature, top-k, top-p, typical, contrastive)
    - Stopping criteria (max_length, max_new_tokens, EOS token)
    - Logits processors (repetition penalty, no repeat n-gram, etc.)

    Incremental Decoding Support
    ----------------------------
    The model supports efficient autoregressive generation using a compressed
    KV‑cache. The cache structure depends on config.use_rope:
    - If use_rope=False (NoPE): (global_state, latent_kv) per layer
    - If use_rope=True (RoPE): (global_state, latent_kv, rope_k) per layer

    Usage:
        # First forward (prompt processing)
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # Generate text (uses KV-cache automatically)
        generated = model.generate(input_ids, max_new_tokens=100)

        # Or manual incremental decoding
        next_token = sample(outputs.logits[:, -1, :])
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)

    Example::
        >>> # NoPE model (legacy)
        >>> config = KilatConfig(vocab_size=50257, n_embd=512, n_head=8, n_layer=8, use_rope=False)
        >>> model = KilatTransformer(config)
        >>>
        >>> # Load pretrained model (auto-detects RoPE/NoPE)
        >>> model = KilatTransformer.from_pretrained("AiRukua/BabyKilat")
        >>>
        >>> input_ids = torch.randint(0, 50257, (2, 128))
        >>> out = model(input_ids, labels=input_ids.clone())
        >>> print(out.loss)
    """

    def __init__(self, config: KilatConfig):
        if isinstance(config, MainConfig):
            config = config.model

        # Ensure config is either KilatConfig or MainConfig
        if not isinstance(config, KilatConfig):
            raise TypeError(
                f"Parameter config must be a KilatConfig or MainConfig instance, "
                f"got {type(config)}. If you want to load a pretrained model, "
                f"use `KilatTransformer.from_pretrained(...)`."
            )

        # ===== BACKWARD COMPATIBILITY: Inject missing RoPE fields =====
        # Older configs (NoPE models) don't have these fields
        if not hasattr(config, 'use_rope'):
            config.use_rope = False
        if not hasattr(config, 'rope_head_dim'):
            config.rope_head_dim = None
        if not hasattr(config, 'rope_base'):
            config.rope_base = 10000.0
        if not hasattr(config, 'ff_mult'):
            config.ff_mult = 8 / 3
        # ===============================================================

        super().__init__(config)
        self.config = config

        # Token embeddings: maps token IDs → dense vectors.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Embedding dropout
        self.drop = (
            nn.Dropout(config.embd_drop) if config.embd_drop > 0 else nn.Identity()
        )

        # Stack of transformer blocks with RoPE support
        self.layers = nn.ModuleList([
            Block(
                n_embd=config.n_embd,
                n_head=config.n_head,
                recall_ratio=config.recall_ratio,
                latent_dim=config.latent_dim,
                attn_drop=config.attn_drop,
                ffn_mode=config.ffn_mode,
                ff_mult=config.ff_mult,
                ffn_dropout=config.ffn_dropout,
                num_experts=config.num_experts,
                active_experts=config.active_experts,
                aux_loss_coef=config.aux_loss_coef,
                resid_drop=config.resid_drop,
                use_rope=config.use_rope,
                rope_head_dim=config.rope_head_dim,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_layer)
        ])

        # Final normalisation + LM head
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight
        self._tied_weights_keys = {"lm_head.weight": "wte.weight"}

        # Initialise weights
        self.post_init()

    def _tie_weights(self):
        """
        Tie the weights between input embeddings and output embeddings.

        This method is called automatically by Hugging Face's `from_pretrained`
        after loading the weights. It ensures weight tying is restored even if
        the checkpoint doesn't contain lm_head.weight.
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

    @classmethod
    def _detect_rope_from_weights(cls, pretrained_path: Path) -> bool:
        """
        Detect if model uses RoPE by checking for rope-specific keys in weights.

        Looks for 'q_rope_proj.weight' or 'k_rope_proj.weight' in any weight file.
        """
        # Try safetensors
        weights_path = pretrained_path / SAFE_WEIGHTS_NAME
        if weights_path.exists() and _SAFETENSORS_AVAILABLE:
            try:
                weights = _st_load(str(weights_path))
                for key in weights.keys():
                    if 'q_rope_proj' in key or 'k_rope_proj' in key:
                        return True
            except Exception:
                pass

        # Try pickle format
        weights_path = pretrained_path / WEIGHTS_NAME
        if weights_path.exists():
            try:
                weights = torch.load(weights_path, map_location='cpu', weights_only=False)
                for key in weights.keys():
                    if 'q_rope_proj' in key or 'k_rope_proj' in key:
                        return True
            except Exception:
                pass

        # Try sharded formats
        for index_name in [SAFE_WEIGHTS_INDEX, WEIGHTS_INDEX_NAME]:
            index_path = pretrained_path / index_name
            if index_path.exists():
                try:
                    with open(index_path, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    for key in index.get('weight_map', {}).keys():
                        if 'q_rope_proj' in key or 'k_rope_proj' in key:
                            return True
                except Exception:
                    pass

        return False

    @classmethod
    def _load_state_dict_from_path(cls, pretrained_path: Path) -> dict:
        """
        Load state dict from disk, handling both single-file and sharded formats.
        """
        # Try safetensors single file
        weights_path = pretrained_path / SAFE_WEIGHTS_NAME
        if weights_path.exists() and _SAFETENSORS_AVAILABLE:
            return _st_load(str(weights_path))

        # Try pickle single file
        weights_path = pretrained_path / WEIGHTS_NAME
        if weights_path.exists():
            return torch.load(weights_path, map_location='cpu', weights_only=False)

        # Try sharded safetensors
        index_path = pretrained_path / SAFE_WEIGHTS_INDEX
        if index_path.exists() and _SAFETENSORS_AVAILABLE:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard_name in shard_files:
                shard_path = pretrained_path / shard_name
                shard_dict = _st_load(str(shard_path))
                state_dict.update(shard_dict)
            return state_dict

        # Try sharded pickle
        index_path = pretrained_path / WEIGHTS_INDEX_NAME
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard_name in shard_files:
                shard_path = pretrained_path / shard_name
                shard_dict = torch.load(shard_path, map_location='cpu', weights_only=False)
                state_dict.update(shard_dict)
            return state_dict

        raise FileNotFoundError(f"No weights found at {pretrained_path}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a pretrained KilatTransformer model with automatic:
        1. Config loading from checkpoint
        2. RoPE vs NoPE detection (backward compatibility)
        3. Weight tying restoration

        This method properly reads the config.json from the checkpoint and
        builds the model with the correct dimensions (n_embd, n_layer, etc.)

        Parameters
        ----------
        pretrained_model_name_or_path : str or Path
            Hugging Face model ID or local path.
        *args, **kwargs
            Additional arguments passed to the parent from_pretrained.

        Returns
        -------
        KilatTransformer
            Loaded model with proper configuration.
        """
        # Resolve path (download from Hub if needed)
        pretrained_path = Path(pretrained_model_name_or_path)

        if not pretrained_path.exists():
            try:
                from huggingface_hub import snapshot_download
                logger.info(f"Downloading from Hugging Face Hub: {pretrained_model_name_or_path}")
                pretrained_path = Path(
                    snapshot_download(
                        str(pretrained_model_name_or_path),
                        local_files_only=kwargs.get('local_files_only', False),
                    )
                )
            except ImportError:
                raise ImportError(
                    f"Path '{pretrained_path}' not found locally, "
                    "and huggingface_hub not installed. "
                    "Install with: pip install huggingface_hub"
                )

        # ===== STEP 1: Load config from checkpoint =====
        config_path = pretrained_path / CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        logger.info(f"Loaded config from checkpoint: n_embd={config_dict.get('n_embd')}, "
                   f"n_layer={config_dict.get('n_layer')}, vocab_size={config_dict.get('vocab_size')}")

        # ===== STEP 2: Detect RoPE from weights =====
        detected_rope = cls._detect_rope_from_weights(pretrained_path)
        logger.info(f"Detected from weights: use_rope={detected_rope}")

        # ===== STEP 3: Inject missing fields =====
        config_dict['use_rope'] = detected_rope
        if 'rope_head_dim' not in config_dict:
            n_embd = config_dict.get('n_embd', 512)
            n_head = config_dict.get('n_head', 8)
            head_dim = n_embd // n_head
            config_dict['rope_head_dim'] = head_dim // 2
        if 'rope_base' not in config_dict:
            config_dict['rope_base'] = 10000.0
        if 'ff_mult' not in config_dict:
            hidden_dim = config_dict.get('ffn_hidden_dim', None)
            if hidden_dim:
                n_embd = config_dict.get('n_embd', 512)
                config_dict['ff_mult'] = hidden_dim / n_embd
            else:
                config_dict['ff_mult'] = 8 / 3

        # ===== STEP 4: Create config object =====
        config = cls.config_class.from_dict(config_dict)

        # ===== STEP 5: Build model with the config from checkpoint =====
        model = cls(config)

        # ===== STEP 6: Load state_dict from disk =====
        state_dict = cls._load_state_dict_from_path(pretrained_path)

        # ===== STEP 7: Load with strict=False =====
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            # Filter out tied weights from missing keys (they're expected)
            tied_keys = set(model._tied_weights_keys.values())
            missing_filtered = [k for k in missing if k not in tied_keys]
            if missing_filtered:
                logger.warning(f"Missing keys (may affect functionality): {missing_filtered[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {unexpected[:10]}...")

        # ===== STEP 8: Restore weight tying =====
        if config.tie_word_embeddings:
            if model.lm_head.weight is not model.wte.weight:
                warnings.warn(
                    "lm_head.weight not tied to wte.weight. "
                    "Automatically restoring weight tying...",
                    UserWarning,
                    stacklevel=2
                )
                model.lm_head.weight = model.wte.weight
                logger.info("Weight tying restored successfully")

        # ===== STEP 9: Log final status =====
        if config.use_rope:
            logger.info("✅ Model loaded in RoPE mode (positional encoding ENABLED)")
        else:
            logger.info("✅ Model loaded in NoPE mode (positional encoding DISABLED)")

        return model

    def get_input_embeddings(self) -> nn.Embedding:
        """Return input embedding layer for HF generation pipeline compatibility."""
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        """Replace input embeddings while preserving weight tying."""
        self.wte = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        """Return output embedding layer (LM head) for HF compatibility."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Replace output embeddings while preserving weight tying."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.

        Parameters
        ----------
        input_ids : torch.Tensor
            (B, N) LongTensor of token indices.
        labels : Optional[torch.Tensor]
            (B, N) LongTensor for loss computation.
        return_dict : Optional[bool]
            Whether to return CausalLMOutputWithPast or a tuple.
        past_key_values : Optional[Tuple]
            Caches from previous forward calls for incremental decoding.
        use_cache : Optional[bool]
            If True, returns `past_key_values` for incremental decoding.
        **kwargs : dict
            Absorbs extra arguments from HF Trainer.

        Returns
        -------
        Union[Tuple, CausalLMOutputWithPast]
            Model output with loss, logits, and past_key_values if use_cache=True.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        validate_tensor_rank(input_ids, 2, "input_ids")

        # Embed and apply dropout
        x = self.drop(self.wte(input_ids))

        # Prepare per‑layer cache list
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer
        else:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError(
                    f"past_key_values length ({len(past_key_values)}) does not match "
                    f"number of layers ({self.config.n_layer})"
                )

        # Pass through transformer blocks
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        has_moe = False
        present_key_values = []

        for i, block in enumerate(self.layers):
            x, layer_aux_loss, layer_present = block(
                x,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss
                has_moe = True
            if use_cache:
                present_key_values.append(layer_present)

        if use_cache:
            present_key_values = tuple(present_key_values)
        else:
            present_key_values = None

        # Final norm + project to vocabulary
        x = self.ln_f(x)
        logits = self.lm_head(x)
        validate_finite_tensor(logits, "Logits Output")

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if has_moe:
                loss = loss + total_aux_loss

        # Return format
        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (present_key_values,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=None,
            attentions=None,
        )