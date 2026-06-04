import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import KilatAttention
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

    def forward(self, x):
        # Compute RMS: sqrt(mean(x^2) + eps)
        # keepdim=True preserves the last dimension for broadcasting,
        # allowing division of [B, N, D] by [B, N, 1].
        # Mean over dim=-1 computes the average squared value across
        # the feature dimension for each position independently.
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale: equivalent to x * (weight / rms)
        # The weight is broadcast along batch and sequence dimensions.
        return x / rms * self.weight

class Block(nn.Module):
    """
    Pre‑norm transformer block with fused attention and RMSNorm.

    This block creates a :class:`KilatAttention` module internally and a
    :class:`FeedForward` module (dense SwiGLU or Mixture‑of‑Experts) and
    applies the standard residual scheme:

        x = x + attn(rms1(x))
        x = x + ffn(rms2(x))

    When the FFN is in MoE mode, an auxiliary load‑balancing loss is returned.

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

    5. **Auxiliary loss return**: For MoE FFNs, the block returns a tuple
       ``(output, aux_loss)``. The auxiliary loss must be accumulated across
       all blocks and added to the main loss. Returning it here allows the
       trainer to collect it without the block needing to know about the
       training loop. For dense FFNs, returns ``None`` to indicate no
       auxiliary loss — the trainer can sum only the non-None values.

    Normalization layers are **RMSNorm** (not LayerNorm), which is more
    memory‑efficient and widely adopted in recent LLMs.

    Example usage (dense FFN)::

        >>> block = Block(n_embd=512, n_head=8, ffn_mode='dense')
        >>> x = torch.randn(2, 128, 512)
        >>> out, loss = block(x)
        >>> print(out.shape)   # torch.Size([2, 128, 512])
        >>> print(loss)        # None

    Example usage (MoE FFN)::

        >>> block = Block(n_embd=512, n_head=8, ffn_mode='moe',
        ...               num_experts=8, active_experts=2, aux_loss_coef=0.01)
        >>> x = torch.randn(2, 128, 512)
        >>> out, loss = block(x)
        >>> print(out.shape)   # torch.Size([2, 128, 512])
        >>> print(loss)        # scalar tensor
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        # ---------- Attention parameters ----------
        recall_ratio: float = 0.5,
        latent_dim: Optional[int] = None,
        attn_drop: float = 0.0,
        # ---------- Feed‑forward parameters ----------
        ffn_mode: str = "dense",
        ff_mult: float = 8 / 3,
        ffn_dropout: float = 0.0,
        num_experts: int = 8,
        active_experts: int = 2,
        aux_loss_coef: float = 0.01,
        # ---------- Residual dropout ----------
        resid_drop: float = 0.0,
    ):
        """
        Parameters
        ----------
        n_embd : int
            Embedding dimension (model width).
        n_head : int
            Number of attention heads (must divide n_embd evenly).
        recall_ratio : float
            Fraction of heads used for latent MLA path in KilatAttention.
            Range: [0, 1]. Default 0.5 balances precision and efficiency.
        latent_dim : Optional[int]
            Latent dimension for MLA projections. None uses n_embd // 4.
        attn_drop : float
            Dropout inside attention (applied during training only).
        ffn_mode : str
            'dense' for SwiGLU FFN, 'moe' for Mixture of Experts.
        ff_mult : float
            Expansion factor for FFN hidden layer. Default 8/3 ≈ 2.67x
            compensates for SwiGLU's gating (which halves the effective
            hidden size) to maintain ~4x expansion like standard FFNs.
        ffn_dropout : float
            Dropout inside FFN.
        num_experts : int
            (MoE only) Total number of experts.
        active_experts : int
            (MoE only) Top-K experts activated per token.
        aux_loss_coef : float
            (MoE only) Coefficient for load-balancing auxiliary loss.
        resid_drop : float
            Dropout applied after the FFN residual branch before adding
            to the main residual stream. 0.0 disables (uses nn.Identity).
        """
        super().__init__()

        # Pre‑norm layers: RMSNorm applied before each sublayer.
        # Each gets its own learnable scale parameter since attention
        # and FFN operate on different aspects of the representation
        # and may benefit from different input magnitudes.
        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)

        # Fused attention module: combines global decay (linear complexity)
        # and latent MLA (compressed KV-cache) attention pathways.
        # See KilatAttention docstring for architecture details.
        self.attn = KilatAttention(
            n_embd=n_embd,
            n_head=n_head,
            recall_ratio=recall_ratio,
            latent_dim=latent_dim,
            attn_drop=attn_drop,
        )

        # Feed‑forward module (dense SwiGLU or MoE with SwiGLU experts).
        # SwiGLU is used instead of ReLU/GELU because it consistently
        # outperforms in transformer architectures (Shazeer, 2020).
        self.mlp = FeedForward(
            dim=n_embd,
            mode=ffn_mode,
            ff_mult=ff_mult,
            dropout=ffn_dropout,
            num_experts=num_experts,
            active_experts=active_experts,
            aux_loss_coef=aux_loss_coef,
        )

        # Residual dropout on the FFN branch only.
        # Using nn.Identity when dropout=0 avoids the overhead of
        # calling nn.Dropout(p=0) which still does a no-op check.
        self.resid_drop = nn.Dropout(resid_drop) if resid_drop > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with pre-norm and residual connections.

        The pre-norm pattern (norm before sublayer, add residual after)
        ensures that the residual stream carries "clean" representations
        while the sublayers work on normalized inputs. This is the reverse
        of the original Transformer's post-norm design and has become
        standard for large model training.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, N, n_embd)`` where:
            B = batch size, N = sequence length, n_embd = hidden dimension.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - **output**: Tensor of shape ``(B, N, n_embd)``.
            - **aux_loss**: Scalar tensor if FFN is MoE (should be summed
              across all blocks and added to main loss), else ``None``.
        """
        # Attention residual path:
        # 1. Apply RMSNorm (pre-norm)
        # 2. Run attention (includes both global decay and latent MLA paths)
        # 3. Add to residual stream (x = x + attention_output)
        # The residual connection preserves the original signal, while
        # attention adds context-dependent information.
        x = x + self.attn(self.rms1(x))

        # FFN residual path:
        # 1. Apply RMSNorm (pre-norm, separate from attention norm)
        # 2. Run feed-forward network (dense or MoE)
        # 3. Apply residual dropout (if configured)
        # 4. Add to residual stream
        # The FFN transforms each position independently, adding
        # position-wise processing capacity to the block.
        ffn_out, aux_loss = self.mlp(self.rms2(x))
        x = x + self.resid_drop(ffn_out)

        return x, aux_loss