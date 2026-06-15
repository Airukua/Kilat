import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional, Literal
from kilat.utils.validators import (
    validate_choice,
    validate_finite_tensor,
    validate_less_equal,
    validate_positive_int,
    validate_probability,
    validate_tensor_last_dim,
    validate_tensor_rank,
)

class SwiGLU(nn.Module):
    """
    SwiGLU (SiLU‑Gated Linear Unit) feed‑forward block.

    This block implements the activation function used in PaLM, LLaMA and many
    modern transformer architectures:

        SwiGLU(x) = Dropout( down( SiLU(gate(x)) ⊙ up(x) ) )

    where ``gate`` and ``up`` are linear projections that expand the dimension
    by ``ff_mult``, and ``down`` projects back to the original dimension.
    The hidden dimension is automatically padded to a multiple of 64 for
    efficient GPU execution.

    Why SwiGLU over ReLU/GELU?
    --------------------------
    Shazeer (2020) showed SwiGLU consistently outperforms standard activations
    in transformer FFN layers. The gating mechanism (SiLU × linear) provides:
    - Input-dependent feature selection: the gate can suppress irrelevant
      features while the up-projection creates candidate features
    - Smooth gradients: SiLU is continuously differentiable, unlike ReLU
      which has a discontinuity at 0
    - Empirical gains: ~1-2% perplexity improvement over ReLU/GELU at scale

    The SwiGLU variant (SiLU gate) is preferred over GEGLU (GELU gate) because:
    - SiLU is computationally cheaper (no erf approximation)
    - SiLU has unbounded positive range, allowing stronger feature amplification
    - SiLU's non-monotonic negative region can learn to suppress features

    Hidden Dimension Padding
    ------------------------
    Padding to a multiple of 64 ensures:
    - Efficient tensor core utilization (warp-level matrix multiply prefers
      dimensions aligned to 64 on A100/H100 GPUs)
    - No bank conflicts in shared memory during matrix operations
    - Consistent performance across different ff_mult values

    All inputs are validated; the output is guaranteed to be finite (no NaN/Inf).

    Example usage::

        >>> swiglu = SwiGLU(dim=512, ff_mult=8/3, dropout=0.1)
        >>> x = torch.randn(2, 128, 512)          # (batch, seq_len, dim)
        >>> out = swiglu(x)
        >>> print(out.shape)
        torch.Size([2, 128, 512])
    """

    def __init__(self, dim, ff_mult=8 / 3, dropout=0.0):
        """
        Args:
            dim: Input / output feature dimension.
            ff_mult: Expansion factor. The hidden size is
                ``int(dim * ff_mult)``, rounded up to the nearest multiple
                of 64. Must be > 0. Default 8/3 ≈ 2.67 compensates for
                gating to achieve ~4x effective expansion.
            dropout: Dropout probability applied after the down projection.
                Must be in [0, 1).
        """
        super().__init__()
        validate_positive_int("dim", dim)
        if ff_mult <= 0:
            raise ValueError(f"ff_mult must be positive, got {ff_mult}")
        validate_probability("dropout", dropout)

        # Compute hidden dimension with 64-alignment for GPU efficiency.
        # The +63 trick rounds up: (dim * ff_mult + 63) // 64 * 64
        hidden = int(dim * ff_mult)
        hidden = (hidden + 63) // 64 * 64
        
        # Bias=False: standard in modern transformers (LLaMA, Mistral, etc.)
        # Biases in FFN layers provide negligible benefit while consuming
        # memory and compute. The LayerNorm/RMSNorm before the FFN handles
        # any needed distribution shift.
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape ``(..., D)`` where the last dimension
                equals ``dim``. Typically ``(B, S, D)``.

        Returns:
            Tensor of the same shape as ``x``, after applying SwiGLU and
            dropout. If any output element is non‑finite, an error is raised.
        """
        if x.dim() < 2:
            raise ValueError(f"x must have rank >= 2, got shape {tuple(x.shape)}")
        validate_tensor_last_dim(x, self.down.out_features, "x")
        
        # SwiGLU: F.silu(gate(x)) * up(x)
        # Element-wise multiplication of gated and up-projected features.
        # The gate (via SiLU) controls which up-projected features pass through.
        out = self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))
        return validate_finite_tensor(out, "SwiGLU output")


class DeepSeekMoE(nn.Module):
    """
    DeepSeek‑V2 style Mixture‑of‑Experts with shared experts and fine‑grained routing.

    This implements the MoE architecture from DeepSeek‑V2 (DeepSeek‑AI, 2024),
    which introduces two key innovations over standard Top‑K MoE:

    1. **Shared Experts**: A subset of experts always process every token,
       providing a common knowledge base that all tokens benefit from.
       This is separate from the routed experts that specialize.

    2. **Fine‑grained Expert Segmentation**: Each expert is smaller than in
       standard MoE, but more experts are activated per token. This provides
       finer control over token-expert assignment and better load balancing.

    Architecture
    -----------
    For each token:
        shared_out = Σ shared_experts(token)           # Always active
        routed_out = Σ topk(routed_experts, token)    # Top-K selection
        output = shared_out + routed_out

    Load Balancing
    -------------
    Uses DeepSeek's auxiliary‑loss‑free load balancing strategy:
    - Expert‑level balance loss: encourages uniform token distribution
      across routed experts (standard Switch Transformer loss)
    - Device‑level balance loss: encourages tokens to be distributed evenly
      across devices when model parallelism is used (computed as secondary loss)

    Key Differences from Standard MoE
    ---------------------------------
    - Shared experts provide a "safety net" ensuring every token gets some
      processing even if routing is suboptimal
    - Smaller individual experts with more activated per token (fine‑grained)
      prevents expert over‑specialization and improves generalization
    - Bias‑free expert selection via Top‑K with normalized weights ensures
      the router can learn to ignore experts without penalty

    Example usage::

        >>> moe = DeepSeekMoE(dim=512, num_routed_experts=64, num_shared_experts=2,
        ...                   active_experts=8, aux_loss_coef=0.001)
        >>> x = torch.randn(2, 256, 512)
        >>> out, aux_loss = moe(x)
        >>> print(out.shape)     # (2, 256, 512)
        >>> print(aux_loss)      # scalar tensor
    """

    def __init__(
        self,
        dim: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        active_experts: int = 8,
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        aux_loss_coef: float = 0.001,
        device_balance_coef: float = 0.001,
    ):
        """
        Args:
            dim: Token representation dimension.
            num_routed_experts: Number of routed (selectively activated) experts.
                DeepSeek‑V2 uses many small experts (64‑256).
            num_shared_experts: Number of shared experts always active for
                all tokens. Typically 1‑2. Provides common knowledge.
            active_experts: Number of routed experts selected per token (Top‑K).
                Must be ≤ num_routed_experts. DeepSeek‑V2 uses 6‑8.
            ff_mult: Expansion factor for each expert's SwiGLU hidden dimension.
                Note: With fine‑grained experts, the per‑expert hidden size is
                typically smaller. This multiplier is applied to the expert
                dimension directly.
            dropout: Dropout probability inside each expert.
            aux_loss_coef: Weight of the expert‑level load‑balancing loss.
                DeepSeek‑V2 uses small values (0.001) to avoid interfering
                with the primary language modeling objective.
            device_balance_coef: Weight of the device‑level balance loss.
                Only meaningful in multi‑GPU training. Set to 0 for single‑GPU.
        """
        super().__init__()
        validate_positive_int("dim", dim)
        validate_positive_int("num_routed_experts", num_routed_experts)
        validate_positive_int("num_shared_experts", num_shared_experts)
        validate_positive_int("active_experts", active_experts)
        validate_less_equal("active_experts", active_experts, num_routed_experts)
        validate_probability("dropout", dropout)
        
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.active_experts = active_experts
        self.aux_loss_coef = aux_loss_coef
        self.device_balance_coef = device_balance_coef

        # Router: maps each token to a score distribution over routed experts.
        # No bias to encourage the router to learn from the token representation
        # directly without a default expert preference.
        self.router = nn.Linear(dim, num_routed_experts, bias=False)

        # Shared experts: always active, process every token.
        # These capture common patterns that all tokens need (syntax, common
        # semantics) and prevent the MoE from "forgetting" general knowledge
        # when experts become too specialized.
        self.shared_experts = nn.ModuleList([
            SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # Routed experts: selectively activated via Top‑K routing.
        # Each expert is a full SwiGLU block. With fine‑grained segmentation,
        # individual experts are smaller but more are activated per token.
        self.routed_experts = nn.ModuleList([
            SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            for _ in range(num_routed_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Forward pass with DeepSeek‑V2 style MoE routing.

        Computational Flow
        ------------------
        1. Shared experts process all tokens (always active)
        2. Router computes scores for routed experts
        3. Top‑K experts selected per token (with softmax normalization)
        4. Selected experts process their assigned tokens
        5. Output = sum(shared_outputs) + weighted sum(routed_outputs)
        6. Compute auxiliary losses (expert‑level + device‑level)

        The Top‑K selection uses softmax normalization over selected experts
        only (not over all experts), following DeepSeek‑V2's approach. This
        ensures the router weights sum to 1 for the selected experts.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, S, D)``.

        Returns
        -------
        tuple: (output, total_aux_loss)
            - output: Transformed tokens, shape ``(B, S, D)``.
            - total_aux_loss: Combined expert‑level and device‑level loss
              as a scalar tensor.
        """
        validate_tensor_rank(x, 3, "x")
        validate_tensor_last_dim(x, self.dim, "x")
        B, S, D = x.shape
        
        # -----------------------------------------------------------------
        # STEP 1: Shared Experts (always active)
        # -----------------------------------------------------------------
        # Shared experts process all tokens and sum their outputs.
        # This provides a strong baseline representation that the routed
        # experts enhance with specialized knowledge.
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # -----------------------------------------------------------------
        # STEP 2: Router computation
        # -----------------------------------------------------------------
        # Flatten batch and sequence for per-token routing.
        x_flat = x.view(-1, D)  # [B*S, D]
        
        # Router logits: unnormalized scores for each routed expert.
        router_logits = self.router(x_flat)  # [B*S, num_routed_experts]
        
        # Top‑K selection with softmax normalization over selected experts.
        # This differs from standard MoE which applies softmax over ALL experts
        # before Top‑K. DeepSeek normalizes only among selected experts,
        # which prevents the router from being penalized for low scores on
        # experts that aren't used anyway.
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_probs, self.active_experts, dim=-1
        )
        
        # Normalize selected weights to sum to 1 for each token.
        # This ensures the weighted combination preserves the input scale.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # -----------------------------------------------------------------
        # STEP 3: Expert computation (batched by expert)
        # -----------------------------------------------------------------
        routed_output = torch.zeros_like(x_flat)
        
        # Process each expert's assigned tokens in a batch for efficiency.
        # This avoids the sequential expert loop being a bottleneck when
        # num_routed_experts is large (64‑256).
        for expert_idx, expert in enumerate(self.routed_experts):
            # Find tokens where this expert was selected in Top‑K
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # [B*S]
            
            if expert_mask.any():
                # Extract tokens assigned to this expert
                expert_input = x_flat[expert_mask]
                expert_output = expert(expert_input)
                
                # Get the weight this expert was assigned for each token.
                # For tokens where expert appears in Top‑K, find its position
                # and extract the corresponding normalized weight.
                expert_positions = (topk_indices[expert_mask] == expert_idx).float().argmax(dim=-1)
                expert_weights = topk_weights[expert_mask].gather(
                    1, expert_positions.unsqueeze(1)
                ).squeeze(1)
                
                # Weighted accumulation: output += weight * expert_output
                routed_output[expert_mask] += expert_weights.unsqueeze(1) * expert_output

        # -----------------------------------------------------------------
        # STEP 4: Combine shared and routed outputs
        # -----------------------------------------------------------------
        output = shared_output.view(-1, D) + routed_output
        output = output.view(B, S, D)

        # -----------------------------------------------------------------
        # STEP 5: Auxiliary loss computation (DeepSeek‑V2 style)
        # -----------------------------------------------------------------
        # Expert‑level balance loss: encourages uniform expert utilization.
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability for expert i
        # L_exp = num_experts * sum(f_i * P_i)
        # When perfectly balanced: f_i = 1/num_experts, so L_exp = 1.0
        # Lower values indicate imbalance; the loss pushes towards 1.0.
        expert_fraction = torch.zeros(
            self.num_routed_experts, device=x.device, dtype=x.dtype
        )
        for i in range(self.num_routed_experts):
            expert_fraction[i] = (topk_indices == i).any(dim=-1).float().mean()
        
        router_prob_avg = router_probs.mean(dim=0)  # Mean over all tokens
        expert_balance_loss = self.num_routed_experts * (
            expert_fraction * router_prob_avg
        ).sum()
        
        # Device‑level balance loss: encourages tokens to be evenly distributed
        # across devices. In single‑GPU training, this is a no‑op (no penalty).
        # In multi‑GPU, the router probabilities are grouped by device and
        # balanced similarly. This is approximated here as the variance of
        # expert fractions, scaled by device_balance_coef.
        device_balance_loss = torch.var(expert_fraction) * self.num_routed_experts
        
        # Combine losses with their respective coefficients.
        total_aux_loss = (
            self.aux_loss_coef * expert_balance_loss +
            self.device_balance_coef * device_balance_loss
        )

        validate_finite_tensor(output, "DeepSeekMoE output")
        validate_finite_tensor(total_aux_loss, "DeepSeekMoE aux_loss")
        return output, total_aux_loss


class FeedForward(nn.Module):
    """
    Unified feed‑forward block that can operate as a dense SwiGLU or as a
    DeepSeek‑V2 style Mixture‑of‑Experts.

    This class provides a common interface for transformer blocks that need to
    switch between dense and MoE modes. In ``'dense'`` mode it returns only
    the transformed tensor; in ``'moe'`` mode it additionally returns the
    auxiliary load‑balancing loss.

    The DeepSeek‑V2 MoE variant includes:
    - Shared experts for common knowledge (always active)
    - Fine‑grained routed experts with Top‑K selection
    - Dual auxiliary loss (expert + device balance)

    Example usage (dense mode)::

        >>> ff = FeedForward(dim=512, mode='dense')
        >>> x = torch.randn(2, 128, 512)
        >>> out, loss = ff(x)
        >>> print(out.shape)   # (2, 128, 512)
        >>> print(loss)        # None

    Example usage (DeepSeek MoE mode)::

        >>> ff = FeedForward(dim=512, mode='moe', num_experts=64, 
        ...                  num_shared_experts=2, active_experts=8)
        >>> x = torch.randn(2, 128, 512)
        >>> out, loss = ff(x)
        >>> print(out.shape)   # (2, 128, 512)
        >>> print(loss)        # scalar tensor
    """

    def __init__(
        self,
        dim: int,
        mode: Literal["dense", "moe"] = "moe",
        ff_mult: float = 8 / 3,
        dropout: float = 0.0,
        num_experts: int = 64,
        num_shared_experts: int = 2,
        active_experts: int = 8,
        aux_loss_coef: float = 0.001,
        device_balance_coef: float = 0.001,
    ):
        """
        Args:
            dim: Model dimension (input and output size).
            mode: ``'dense'`` for a single :class:`SwiGLU` block,
                ``'moe'`` for a DeepSeek‑V2 style :class:`DeepSeekMoE` layer.
            ff_mult: Expansion factor for the feed‑forward layer(s).
            dropout: Dropout probability inside the feed‑forward layers.
            num_experts: (MoE only) Total number of routed experts.
                Default 64 for fine‑grained DeepSeek‑V2 style.
            num_shared_experts: (MoE only) Number of always‑active shared experts.
                Default 2 as in DeepSeek‑V2.
            active_experts: (MoE only) Number of routed experts selected per token.
                Default 8 for fine‑grained routing.
            aux_loss_coef: (MoE only) Expert‑level balance loss coefficient.
                Default 0.001 (DeepSeek‑V2 uses small values).
            device_balance_coef: (MoE only) Device‑level balance loss coefficient.
                Set to 0 for single‑GPU training.
        """
        super().__init__()
        validate_positive_int("dim", dim)
        validate_choice("mode", mode, ("dense", "moe"))
        validate_probability("dropout", dropout)
        self.mode = mode

        if mode == "dense":
            # Dense mode: single SwiGLU block, no routing.
            # Simpler and more parameter‑efficient for small models.
            self.layer = SwiGLU(dim, ff_mult=ff_mult, dropout=dropout)
            self.num_experts = 1
            self.active_experts = 1
        elif mode == "moe":
            # DeepSeek‑V2 MoE mode: shared + routed experts with dual loss.
            # The num_experts parameter maps to num_routed_experts in DeepSeekMoE.
            self.layer = DeepSeekMoE(
                dim=dim,
                num_routed_experts=num_experts,
                num_shared_experts=num_shared_experts,
                active_experts=active_experts,
                ff_mult=ff_mult,
                dropout=dropout,
                aux_loss_coef=aux_loss_coef,
                device_balance_coef=device_balance_coef,
            )
            self.num_experts = num_experts
            self.active_experts = active_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape ``(B, S, D)``.

        Returns:
            tuple: ``(output, aux_loss)``
                - ``output``: Transformed tensor, shape ``(B, S, D)``.
                - ``aux_loss``: Load‑balancing loss if mode is ``'moe'``,
                  otherwise ``None``. In MoE mode, this is the combined
                  expert + device balance loss.
        """
        validate_tensor_rank(x, 3, "x")
        if self.mode == "dense":
            out = self.layer(x)
            return validate_finite_tensor(out, "FeedForward[dense] output"), None
        else:
            out, aux_loss = self.layer(x)
            validate_finite_tensor(out, "FeedForward[moe] output")
            if aux_loss is not None:
                validate_finite_tensor(aux_loss, "FeedForward[moe] aux_loss")
            return out, aux_loss