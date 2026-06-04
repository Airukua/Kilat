from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from .blocks import Block, RMSNorm
from utils.config import KilatConfig
from utils.sanity_check import (
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


class KilatTransformerHF(KilatPreTrainedModel):
    """
    Hugging Face‑compatible KilatTransformer for causal language modelling.

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

    Example::
        >>> config = KilatConfig(vocab_size=32000, n_embd=768, n_head=12, n_layer=12)
        >>> model = KilatTransformerHF(config)
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> labels = input_ids.clone()
        >>> out = model(input_ids, labels=labels)
        >>> print(out.loss)       # scalar loss (incl. auxiliary if MoE)
        >>> print(out.logits.shape)  # (2, 128, 32000)
    """

    def __init__(self, config: KilatConfig):
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

        # Required by transformers >= 4.46 for safe_serialization when tensors
        # are shared across modules. Without this, save_pretrained raises:
        # "shared tensors mismatching the transformers base configuration".
        # The set contains the attribute names (not parameter names) that share
        # storage, telling the serialization logic to handle them correctly
        # during safetensor sharding.
        self._dynamic_tied_weights_keys = {"lm_head.weight", "wte.weight"}

        # Initialise weights via Hugging Face post‑init.
        # This calls _init_weights on every submodule, then runs any
        # additional initialization registered by PreTrainedModel.
        self.post_init()

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

        Parameters
        ----------
        input_ids : torch.Tensor
            (B, N) LongTensor of token indices.
        labels : Optional[torch.Tensor]
            (B, N) LongTensor for loss computation. Padding positions
            should use -100 (ignored by cross‑entropy).
        return_dict : Optional[bool]
            Whether to return CausalLMOutputWithPast or a tuple. If None,
            uses self.config.use_return_dict.
        **kwargs : dict
            Absorbs extra arguments from HF Trainer (attention_mask, etc.)
            to prevent TypeError. These are intentionally ignored.

        Returns
        -------
        Union[Tuple, CausalLMOutputWithPast]
            If return_dict=True: CausalLMOutputWithPast with loss and logits.
            If return_dict=False: tuple of (loss, logits) or just (logits,)
            if labels is None.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        validate_tensor_rank(input_ids, 2, "input_ids")

        # Embed and apply dropout
        x = self.drop(self.wte(input_ids))

        # Pass through transformer blocks, accumulating MoE auxiliary loss.
        # The auxiliary loss is a scalar tensor that accumulates across blocks.
        # It starts at 0.0 on the correct device/dtype to avoid any device
        # mismatch when adding layer losses.
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        has_moe = False  # Track if any block actually uses MoE

        for block in self.layers:
            x, layer_aux_loss = block(x)
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss
                has_moe = True

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
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        # CausalLMOutputWithPast: standard HF output dataclass.
        # past_key_values=None is a placeholder — KV‑cache support requires
        # implementing cache tracking for the compressed MLA KV states.
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,   # KV‑cache placeholder for future integration
            hidden_states=None,
            attentions=None,
        )