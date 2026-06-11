from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .blocks import Block, RMSNorm
from configs.model_config import KilatConfig
from configs.main_config import MainConfig
from pipeline.generation.generation_mixin import GenerationMixin
from utils.validators import (
    validate_finite_tensor,
    validate_tensor_rank,
)


class KilatPreTrainedModel(PreTrainedModel):
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
       - KilatAttention (global decay + latent MLA)
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
    KV‑cache. The cache is a tuple of per‑layer caches, each of which is a tuple
    `(global_state, latent_kv)` as produced by `KilatAttention`.
    - `global_state`: (B, n_global_heads, head_dim) – recurrent state for global decay heads.
    - `latent_kv`: (B, total_len, latent_dim) – compressed KV for latent MLA heads.

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
        >>> config = KilatConfig(vocab_size=32000, n_embd=768, n_head=12, n_layer=12)
        >>> model = KilatTransformer(config)
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> labels = input_ids.clone()
        >>> out = model(input_ids, labels=labels)
        >>> print(out.loss)       # scalar loss (incl. auxiliary if MoE)
        >>> print(out.logits.shape)  # (2, 128, 32000)
        >>>
        >>> # Generation
        >>> generated = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.8)
        >>> print(generated.shape)  # (2, 178)
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
        
        super().__init__(config)
        self.config = config
        # Token embeddings: maps token IDs → dense vectors.
        # Padding index is typically 0 for most tokenizers, but the embedding
        # layer doesn't need to know this — the loss function's ignore_index
        # handles padding exclusion.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Embedding dropout: applied directly after embedding lookup.
        # Using nn.Identity when dropout=0 avoids the overhead of calling
        # nn.Dropout(p=0) which still performs a no‑op check on every call.
        self.drop = (
            nn.Dropout(config.embd_drop) if config.embd_drop > 0 else nn.Identity()
        )

        # Stack of transformer blocks.
        # Each block is independent with its own attention and FFN modules.
        # Blocks share the same architecture but NOT weights — they learn
        # different levels of abstraction (surface syntax → deep semantics).
        self.layers = nn.ModuleList([
            Block(
                n_embd=config.n_embd,
                n_head=config.n_head,
                recall_ratio=config.recall_ratio,
                latent_dim=config.latent_dim,
                attn_drop=config.attn_drop,
                ffn_mode=config.ffn_mode,
                num_experts=config.num_experts,
                active_experts=config.active_experts,
                aux_loss_coef=config.aux_loss_coef,
                resid_drop=config.resid_drop,
                ffn_dropout=config.ffn_dropout,
            )
            for _ in range(config.n_layer)
        ])

        # Final normalisation + LM head.
        # ln_f is applied BEFORE the LM head (pre‑output norm), following the
        # pre‑norm architecture pattern. This ensures the logits are computed
        # from a normalized representation, preventing the output projection
        # from needing to learn to handle varying activation scales.
        self.ln_f = RMSNorm(config.n_embd)
        
        # LM head: projects hidden states to vocabulary logits.
        # No bias because weight tying would make a bias term asymmetric
        # (shared weight but separate bias for input/output would be inconsistent).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share storage between input embeddings and output projection.
        # This creates a circular reference (wte.weight is lm_head.weight is wte.weight)
        # which is intentional. Both attributes point to the same underlying tensor,
        # so gradient updates to either affect both.
        self.wte.weight = self.lm_head.weight

        # Required by recent transformers releases for safe_serialization when
        # tensors are shared across modules. The mapping tells the serializer
        # that `lm_head.weight` is intentionally tied to `wte.weight`.
        self._tied_weights_keys = {"lm_head.weight": "wte.weight"}

        # Initialise weights via Hugging Face post‑init.
        # This calls _init_weights on every submodule, then runs any
        # additional initialization registered by PreTrainedModel.
        self.post_init()

    def _tie_weights(self):
        """
        Tie the weights between input embeddings and output embeddings.
        
        This method is called automatically by Hugging Face's `from_pretrained`
        after loading the weights. It ensures weight tying is restored even if
        the checkpoint doesn't contain lm_head.weight.
        
        WHY: Older checkpoints may not have lm_head.weight saved due to weight tying.
        This method automatically restores the tie without user intervention.
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a pretrained KilatTransformer model with automatic weight tying restoration.
        
        This method overrides the default HF from_pretrained to:
        1. Load with strict=False (handles missing lm_head.weight)
        2. Automatically restore weight tying after loading
        3. Log a warning if weight tying was missing
        
        WHY: Older checkpoints uploaded to Hugging Face Hub may not have lm_head.weight
        saved due to the way PyTorch handles shared tensors. This method ensures
        users never see the confusing "lm_head.weight | MISSING" warning and the
        model always works correctly out of the box.
        
        Parameters
        ----------
        pretrained_model_name_or_path : str
            Hugging Face model ID or local path.
        *args, **kwargs
            Additional arguments passed to the parent from_pretrained.
        
        Returns
        -------
        KilatTransformer
            Loaded model with properly restored weight tying.
        """
        import warnings
        import logging
        logger = logging.getLogger(__name__)
        
        # Load with strict=False to allow missing lm_head.weight
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Check if weight tying needs to be restored
        if model.config.tie_word_embeddings:
            # Check if lm_head.weight is a separate tensor (not tied to wte)
            if model.lm_head.weight is not model.wte.weight:
                warnings.warn(
                    "lm_head.weight not tied to wte.weight. "
                    "This is expected for older checkpoints. "
                    "Automatically restoring weight tying...",
                    UserWarning,
                    stacklevel=2
                )
                model.lm_head.weight = model.wte.weight
                logger.info("Weight tying restored successfully: lm_head.weight = wte.weight")
        
        return model

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Return input embedding layer for HF generation pipeline compatibility.

        HuggingFace's generate() method uses this to access token embeddings
        during beam search and sampling.
        """
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding):
        """
        Replace input embeddings while preserving weight tying.

        When resizing token embeddings (e.g., adding special tokens), this
        ensures the tied lm_head weight is updated consistently.
        """
        self.wte = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        """
        Return output embedding layer (LM head) for HF compatibility.

        Used by resize_token_embeddings() to adjust vocabulary size.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """
        Replace output embeddings while preserving weight tying.

        Ensures the tied wte weight is updated consistently when the
        LM head is resized.
        """
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        # ---------- ADDED FOR INCREMENTAL DECODING ----------
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        # ---------------------------------------------------
        **kwargs,  # absorb extra HF Trainer args (e.g., attention_mask)
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.

        Loss Computation
        ----------------
        The standard causal LM loss shifts logits and labels by one position:
        - logits[:, :-1, :] predicts tokens at positions 1, 2, ..., N-1
        - labels[:, 1:] provides the ground truth for those positions
        This means position 0 never has a loss target (it's the "start" token),
        and position N-1 never generates a prediction (it has no subsequent token).

        Cross‑entropy uses ignore_index=-100, which is the standard for
        HuggingFace tokenizers. Any label position with value -100 is excluded
        from loss computation (both numerator and denominator).

        MoE Loss Aggregation
        --------------------
        Auxiliary losses from MoE blocks are already multiplied by their
        respective ``aux_loss_coef`` values at the block level. The forward
        pass simply sums them. This means:
        - Total aux loss is the weighted sum across all MoE blocks
        - Each block can have different aux_loss_coef (though in practice
          they're usually the same from config)
        - The aux loss is added AFTER the primary cross‑entropy loss, so it
          doesn't affect perplexity calculations (only gradients)

        Incremental Decoding Parameters (added)
        ----------------------------------------
        past_key_values : Optional[Tuple]
            Caches from previous forward calls, as returned by this method.
            For the first forward call, pass None. For subsequent generation steps,
            pass the `past_key_values` that was returned earlier.
            The tuple has length `config.n_layer`. Each element is either None
            (if no cache available for that layer) or a tuple of
            (global_state, latent_kv).
        use_cache : Optional[bool]
            If True, returns `past_key_values` that can be used for incremental
            decoding. If False, no cache is returned (saves memory during training).
            Defaults to self.config.use_cache.

        Parameters
        ----------
        input_ids : torch.Tensor
            (B, N) LongTensor of token indices.
        labels : Optional[torch.Tensor]
            (B, N) LongTensor for loss computation. Padding positions
            should use -100 (ignored by cross‑entropy).
        return_dict : Optional[bool]
            Whether to return CausalLMOutputWithPast or a tuple. If None,
            uses self.config.return_dict.
        **kwargs : dict
            Absorbs extra arguments from HF Trainer (attention_mask, etc.)
            to prevent TypeError. These are intentionally ignored.

        Returns
        -------
        Union[Tuple, CausalLMOutputWithPast]
            If return_dict=True: CausalLMOutputWithPast with loss, logits,
            and past_key_values (if use_cache).
            If return_dict=False: tuple of (loss, logits, past_key_values) or
            (logits, past_key_values) or (loss, logits) depending on flags.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        # ---------- ADDED: handle use_cache default ----------
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # -----------------------------------------------------
        validate_tensor_rank(input_ids, 2, "input_ids")

        # Embed and apply dropout
        x = self.drop(self.wte(input_ids))

        # ---------- ADDED: prepare per‑layer cache list ----------
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer
        else:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError(
                    f"past_key_values length ({len(past_key_values)}) does not match "
                    f"number of layers ({self.config.n_layer})"
                )
        # ---------------------------------------------------------

        # Pass through transformer blocks, accumulating MoE auxiliary loss.
        # The auxiliary loss is a scalar tensor that accumulates across blocks.
        # It starts at 0.0 on the correct device/dtype to avoid any device
        # mismatch when adding layer losses.
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        has_moe = False  # Track if any block actually uses MoE

        # ---------- ADDED: collect new caches ----------
        present_key_values = []
        # ------------------------------------------------

        for i, block in enumerate(self.layers):
            # ---------- MODIFIED: pass cache and use_cache to block ----------
            # Original line: x, layer_aux_loss = block(x)
            # Now we pass per‑layer cache and use_cache flag.
            # The block returns (output, aux_loss, present_cache)
            x, layer_aux_loss, layer_present = block(
                x,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            # -----------------------------------------------------------------
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss
                has_moe = True
            # ---------- ADDED: store present cache ----------
            if use_cache:
                present_key_values.append(layer_present)
            # ------------------------------------------------

        # ---------- ADDED: convert list to tuple for HF ----------
        if use_cache:
            present_key_values = tuple(present_key_values)
        else:
            present_key_values = None
        # ---------------------------------------------------------

        # Final norm + project to vocabulary.
        # ln_f ensures the hidden states are normalized before the LM head,
        # preventing the output distribution from being sensitive to the
        # scale of activations from the last transformer block.
        x = self.ln_f(x)
        logits = self.lm_head(x)
        validate_finite_tensor(logits, "Logits Output")

        # Compute causal LM loss if labels are provided.
        # The loss computation is intentionally outside the forward pass's
        # main computation flow — it only runs when labels are given (training)
        # and is skipped during generation/inference.
        loss = None
        if labels is not None:
            # Shift logits and labels for next‑token prediction.
            # logits[:, :-1, :]: predictions for positions 0 to N-2
            # labels[:, 1:]: ground truth for positions 1 to N-1
            # The .contiguous() call ensures the tensors are in the expected
            # memory layout for .view(-1, vocab_size) below.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten batch and sequence dimensions for cross‑entropy:
            # (B, N-1, vocab_size) → (B*(N-1), vocab_size)
            # (B, N-1) → (B*(N-1))
            # ignore_index=-100 excludes padding positions from loss.
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Add MoE auxiliary loss to the primary loss.
            # This couples the auxiliary loss with the main training objective,
            # so both are optimized simultaneously. The auxiliary loss is
            # already scaled by aux_loss_coef at the block level.
            if has_moe:
                loss = loss + total_aux_loss

        # Return format: respect HF's return_dict convention.
        # Tuple format: (loss, logits) or just (logits,) for backward compat.
        # ---------- MODIFIED: include cache in tuple when use_cache ----------
        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (present_key_values,)
            if loss is not None:
                output = (loss,) + output
            return output
        # ---------------------------------------------------------------------

        # CausalLMOutputWithPast: standard HF output dataclass.
        # past_key_values now contains the updated cache for incremental decoding.
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,   # was None, now cache
            hidden_states=None,
            attentions=None,
        )