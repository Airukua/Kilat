import torch
import torch.nn as nn
from typing import Any, Literal, Optional, Tuple
from .attention import KilatAttention, KilatAttentionRoPE
from .ffn import FeedForward

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input tensor along the last dimension by dividing by the
    root mean square of its elements, then applies a learnable per‑dimension
    scale.  Unlike standard LayerNorm, it **omits the mean subtraction step**,
    making it more memory‑efficient and slightly faster, while maintaining
    comparable training stability.  RMSNorm is used in recent large language
    models such as LLaMA, LLaMA 2, and Mistral.

    The forward pass computes:

        out = x / sqrt(mean(x^2) + eps) * weight

    where ``weight`` is a trainable parameter of shape ``normalized_shape``.

    Why RMSNorm over LayerNorm?
    ---------------------------
    1. **Memory efficiency**: No mean subtraction means no need to compute
       or store the mean. This saves one reduction operation and one tensor
       in the computation graph, which matters for large models where
       activation memory dominates.
    2. **Speed**: Fewer operations per normalization step. For large
       transformers with many layers, this accumulates to meaningful
       throughput improvements (~5-10%).
    3. **Empirical equivalence**: Zhang & Sennrich (2019) showed RMSNorm
       performs comparably to LayerNorm for transformer training. LLaMA
       and Mistral models confirmed this at scale.
    4. **Scale-only transformation**: By focusing only on rescaling
       (not re-centering), RMSNorm is more similar to weight normalization
       techniques that have shown success in deep networks.

    The key insight is that for transformer activations, variance control
    (via scaling) is more important than mean centering. The learnable
    weight parameter allows the model to adjust per-dimension magnitude
    as needed.

    Incremental Decoding Compatibility
    ----------------------------------
    RMSNorm is stateless — each position is normalized independently using
    only its own feature values. This means no cache is needed and no changes
    are required for incremental decoding. Unlike attention or recurrent
    layers, normalization layers don't accumulate state across time steps.

    Parameters
    ----------
    normalized_shape : int or tuple
        Shape of the normalization dimension(s). Typically the last
        dimension, e.g., ``(d_model,)`` or just ``d_model``.
    eps : float
        Small constant for numerical stability (default: 1e‑6).
        Added inside the sqrt to prevent division by zero when all
        inputs are exactly zero. 1e-6 is standard and doesn't affect
        the gradient in normal operation.

    Example:
        >>> rms = RMSNorm(512)
        >>> x = torch.randn(2, 128, 512)    # (batch, seq_len, dim)
        >>> y = rms(x)
        >>> print(y.shape)
        torch.Size([2, 128, 512])
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # Learnable scale parameter initialized to 1.0 (identity).
        # Starting at 1 ensures the normalization is initially just
        # the RMS scaling, and the model can learn to amplify or
        # attenuate specific dimensions as training progresses.
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 (preserving float64 for gradcheck) before squaring
        # to prevent FP16 overflow when |x| > 255.9 (65504^0.5).
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x_up = x.to(compute_dtype)
        variance = x_up.pow(2).mean(dim=-1, keepdim=True)
        x_normed = (x_up * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return x_normed * self.weight


class Block(nn.Module):
    """
    Pre‑norm transformer block with fused attention and RMSNorm.

    This block creates a :class:`KilatAttention` or :class:`KilatAttentionRoPE`
    module internally and a :class:`FeedForward` module (dense SwiGLU or
    Mixture‑of‑Experts) and applies the standard residual scheme:

        x = x + attn(rms1(x))
        x = x + ffn(rms2(x))

    When the FFN is in MoE mode, an auxiliary load‑balancing loss is returned.

    Incremental Decoding Support
    ----------------------------
    The block acts as a pass-through for KV-cache: it receives cached states
    from previous steps via ``past_key_values`` and returns updated states
    via ``present_key_values``. The FFN path does NOT participate in caching
    because it's a position-wise operation with no inter-token dependencies.
    
    Cache flow through the block:
        Input:  x, past_key_values
        Step 1: norm_x = rms1(x)                           [stateless]
        Step 2: attn_out, present_kv = attn(norm_x, ...)   [stateful, produces cache]
        Step 3: x = x + attn_out                            [residual, stateless]
        Step 4: ffn_out, aux = mlp(rms2(x))                [stateless]
        Step 5: x = x + dropout(ffn_out)                   [residual, stateless]
        Output: x, aux_loss, present_key_values

    The auxiliary loss from MoE layers is accumulated across blocks during
    training. During incremental decoding, MoE routing operates independently
    on each token — no state tracking needed.

    Design Decisions
    ----------------
    1. **Pre-norm architecture**: Normalization BEFORE each sublayer (attention,
       FFN), NOT after. This is the modern standard (GPT-3+, LLaMA, Mistral)
       because:
       - More stable training: gradients flow through residual paths without
         normalization in the way
       - Allows larger learning rates without divergence
       - The identity-path gradient is unimpeded, preventing vanishing gradients

    2. **Separate RMSNorm per sublayer**: ``rms1`` for attention, ``rms2`` for
       FFN. Each sublayer gets its own learnable scale parameter, allowing the
       model to adjust the magnitude of inputs to attention vs. FFN independently.
       Shared RMSNorm would couple these two very different operations.

    3. **Residual dropout only on FFN branch**: Dropout is applied only after
       the FFN, not after attention. This follows the convention that attention
       benefits less from regularization (the softmax already provides implicit
       regularization via the attention distribution), while the larger-capacity
       FFN benefits more from dropout. If ``resid_drop=0``, this is a no-op
       (nn.Identity) to avoid unnecessary computation.

    4. **No dropout on attention residual**: The attention path already has
       ``attn_drop`` inside the module. Adding residual dropout would double-count
       regularization and potentially harm gradient flow through the critical
       attention pathway.

    5. **Auxiliary loss return**: For MoE FFNs, the block returns the auxiliary
       loss alongside the output. The loss is accumulated across all blocks in
       the main model's forward pass and added to the primary language modeling
       loss. For dense FFNs, returns ``None`` — the trainer can safely sum
       only non-None values. During incremental decoding, the auxiliary loss is
       ignored (MoE routing still operates but without loss computation).

    Normalization layers are **RMSNorm** (not LayerNorm), which is more
    memory‑efficient and widely adopted in recent LLMs.

    Example usage (dense FFN, training)::

        >>> block = Block(n_embd=512, n_head=8, ffn_mode='dense')
        >>> x = torch.randn(2, 128, 512)
        >>> out, aux_loss, cache = block(x)
        >>> print(out.shape)   # torch.Size([2, 128, 512])
        >>> print(aux_loss)    # None
        >>> print(cache)       # None (use_cache=False by default)

    Example usage (MoE FFN, incremental decoding)::

        >>> block = Block(n_embd=512, n_head=8, ffn_mode='moe',
        ...               num_experts=8, active_experts=2, aux_loss_coef=0.01)
        >>> x = torch.randn(2, 1, 512)  # Single new token
        >>> cache = load_cache()  # From previous step
        >>> out, aux_loss, new_cache = block(x, past_key_values=cache, use_cache=True)
        >>> print(out.shape)      # torch.Size([2, 1, 512])
        >>> print(new_cache)      # Updated cache for next step
    """

    def __init__(
        self,
        n_embd: Optional[int] = None,
        n_head: Optional[int] = None,
        # ---------- Attention parameters ----------
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        # ---------- RoPE parameters (optional) ----------
        use_rope: bool = False,
        rope_head_dim: Optional[int] = None,
        rope_base: float = 10000.0,
        # ---------- Feed‑forward parameters ----------
        ffn_mode: Literal["dense", "moe"] = "dense",
        ff_mult: float = 8 / 3,
        ffn_dropout: float = 0.0,
        num_experts: int = 8,
        active_experts: int = 2,
        num_shared_experts: int = 2,
        fine_grained_factor: int = 1,
        aux_loss_coef: float = 0.01,
        device_balance_coef: float = 0.0,
        router_balance_mode: Literal["aux_loss", "bias_free"] = "aux_loss",
        # ---------- Residual dropout & diagnostics ----------
        resid_drop: float = 0.0,
        debug_nan_checks: bool = False,
        config: Optional[Any] = None,
    ):
        """
        Initialise Pre-Norm Transformer Block from either a full ``KilatConfig``
        (single-source-of-truth) or legacy keyword arguments.
        """
        super().__init__()

        if config is not None:
            n_embd = int(config.n_embd)
            n_head = int(config.n_head)
            recall_ratio = float(config.recall_ratio)
            latent_dim = config.latent_dim
            attn_drop = float(config.attn_drop)
            use_rope = bool(getattr(config, "use_rope", False))
            rope_head_dim = getattr(config, "rope_head_dim", None)
            rope_base = float(getattr(config, "rope_base", 10000.0))
            ffn_mode = config.ffn_mode
            ff_mult = float(getattr(config, "ff_mult", 8 / 3))
            ffn_dropout = float(config.ffn_dropout)
            num_experts = int(config.num_experts)
            active_experts = int(config.active_experts)
            num_shared_experts = int(getattr(config, "num_shared_experts", 2))
            fine_grained_factor = int(getattr(config, "fine_grained_factor", 1))
            aux_loss_coef = float(config.aux_loss_coef)
            device_balance_coef = float(getattr(config, "device_balance_coef", 0.0))
            router_balance_mode = getattr(config, "router_balance_mode", "aux_loss")
            resid_drop = float(config.resid_drop)
            debug_nan_checks = bool(getattr(config, "debug_nan_checks", False))

        if n_embd is None or n_head is None:
            raise ValueError("Block requires either `config` or both `n_embd` and `n_head`.")

        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)

        if use_rope:
            self.attn: nn.Module = KilatAttentionRoPE(
                n_embd=n_embd,
                n_head=n_head,
                recall_ratio=recall_ratio,
                latent_dim=latent_dim,
                attn_drop=attn_drop,
                rope_head_dim=rope_head_dim,
                rope_base=rope_base,
            )
        else:
            self.attn = KilatAttention(
                n_embd=n_embd,
                n_head=n_head,
                recall_ratio=recall_ratio,
                latent_dim=latent_dim,
                attn_drop=attn_drop,
            )

        self.mlp = FeedForward(
            dim=n_embd,
            mode=ffn_mode,
            ff_mult=ff_mult,
            dropout=ffn_dropout,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            active_experts=active_experts,
            fine_grained_factor=fine_grained_factor,
            aux_loss_coef=aux_loss_coef,
            device_balance_coef=device_balance_coef,
            router_balance_mode=router_balance_mode,
            debug_nan_checks=debug_nan_checks,
        )

        self.resid_drop = nn.Dropout(resid_drop) if resid_drop > 0 else nn.Identity()
        self.use_rope = use_rope

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """
        Forward pass with pre-norm, residual connections, and KV-cache support.

        The pre-norm pattern (norm before sublayer, add residual after)
        ensures that the residual stream carries "clean" representations
        while the sublayers work on normalized inputs. This is the reverse
        of the original Transformer's post-norm design and has become
        standard for large model training.

        Cache Management
        ---------------
        The block acts as a thin wrapper around KilatAttention for cache
        management. It does NOT modify or inspect the cache — it simply
        passes it through. This separation of concerns keeps the block
        focused on the residual structure while the attention module
        handles all cache logic.

        The FFN path produces an auxiliary loss in MoE mode. During
        incremental decoding, this loss is still produced but typically
        ignored by the caller (who only needs the output and cache).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, n_embd). For incremental decoding,
            N=1 (single new token).
        past_key_values : Optional[Tuple]
            Cached attention states from previous forward pass, or None
            for the first step or full-sequence processing.
        use_cache : bool
            If True, the attention module will return updated cache alongside
            its output. Should be False during training, True during generation.

        Returns
        -------
        Tuple containing:
            - **output**: Tensor of shape (B, N, n_embd). The transformed
              representation after attention, FFN, and residual connections.
            - **aux_loss**: Scalar tensor if FFN is MoE (routed experts active),
              else None. During generation, this is typically ignored.
            - **present_key_values**: Updated cache tuple if use_cache=True,
              else None. Contains the attention module's cache for the next
              incremental step.
        """
        # -----------------------------------------------------------------
        # ATTENTION RESIDUAL PATH
        # -----------------------------------------------------------------
        # The attention sublayer is the ONLY stateful component.
        # 1. Apply RMSNorm (pre-norm) — stateless, works on any sequence length
        # 2. Run attention with cache — produces output and optionally new cache
        # 3. Add to residual stream — preserves identity path for gradient flow
        #
        # The attention module returns either:
        # - use_cache=False: just the output tensor
        # - use_cache=True: (output, cache_tuple)
        attn_out = self.attn(
            self.rms1(x),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        # Unpack attention output based on whether caching is enabled.
        # This conditional maintains backward compatibility: existing training
        # code that doesn't pass use_cache will still work because the default
        # is False, and the attention module returns just the tensor.
        if use_cache:
            attn_output, present_key_values = attn_out
        else:
            attn_output = attn_out
            present_key_values = None
            
        # Residual connection: add attention output to the input.
        # The residual stream (x) accumulates information from all layers,
        # while the attention sublayer contributes context-dependent features.
        x = x + attn_output

        # -----------------------------------------------------------------
        # FFN RESIDUAL PATH
        # -----------------------------------------------------------------
        # The FFN sublayer is stateless — each position is processed
        # independently with no cross-position dependencies. This means:
        # - No cache is needed or produced
        # - Incremental decoding only processes the new token(s)
        # - MoE routing operates independently per token
        #
        # 1. Apply RMSNorm (separate from attention norm)
        # 2. Run feed-forward network (dense or MoE)
        # 3. Apply residual dropout (only during training)
        # 4. Add to residual stream
        ffn_out, aux_loss = self.mlp(self.rms2(x))
        x = x + self.resid_drop(ffn_out)

        # Return output, auxiliary loss, and optionally the updated cache.
        # The cache is None during training to avoid unnecessary memory usage.
        return x, aux_loss, present_key_values