import torch
import triton
import triton.language as tl


@triton.jit
def flash_decay_fwd_kernel(
    v_ptr, lam_ptr, out_ptr,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B, H, N, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Forward kernel for causal exponential decay over a sequence.
    
    Computes a linear recurrence with exponential decay, similar to the
    retention mechanism in RetNet (Retentive Network, Sun et al., 2023)
    or the state update in linear attention variants.

    Mathematical Formulation
    ------------------------
    For each position i in the sequence:
        out[i] = Σ_{j=0}^{i} λ^(i-j) · V[j] / z_i
    
    where:
    - λ ∈ (0, 1] is a head-specific decay factor (lam)
    - λ^(i-j) is the exponential decay weight based on relative distance
    - z_i = Σ_{j=0}^{i} λ^j = (1 - λ^(i+1)) / (1 - λ) is the normalization
      constant (geometric series sum)
    
    The normalization by z_i ensures that the output magnitude doesn't
    explode or vanish with sequence length — without it, earlier positions
    would dominate (if λ < 1) or the sum would grow unbounded (if λ = 1).

    Why This Kernel Exists
    ----------------------
    A naive PyTorch implementation would require O(N²) memory for the
    full (i, j) decay matrix. This kernel computes the same result using
    tiled matrix operations in SRAM, achieving:
    - O(N·D) global memory reads instead of O(N²·D)
    - Fused computation (decay + matmul + normalization) in a single pass
    - No materialization of the full N×N attention/decay matrix
    
    This follows the "FlashAttention" philosophy (Dao et al., 2022):
    tile the computation, keep intermediate results in fast SRAM, and
    only write final results to HBM.

    Kernel Design
    -------------
    The grid is 2D:
    - dim 0 (pid_bh): Iterates over all (batch, head) pairs — each block
      processes one (batch, head) independently.
    - dim 1 (pid_m): Iterates over row tiles of the sequence — each block
      computes BLOCK_M output rows.

    Within each block, an inner loop iterates over column tiles (j axis)
    up to the current row boundary (causal constraint: j ≤ i). This means
    blocks for earlier positions (small pid_m) do less work than later
    blocks, creating a triangular workload. While not perfectly load-balanced,
    this causal structure enables efficient block-sparse computation.

    Analytical Normalization
    -----------------------
    Rather than computing z_i cumulatively during the loop (which would
    require atomic operations or sequential dependencies), we pre-compute
    z_i analytically using the geometric series formula. This is possible
    because the normalization sums over ALL previous positions, not just
    those in the current tile. The pre-computation is O(BLOCK_M) and
    avoids sequential dependencies across tiles.

    Numerical Stability
    ------------------
    - eps = 1e-6 prevents division by zero when λ = 1 and z_i → 0
    - log_lam = log(λ + 1e-9) prevents log(0) when λ = 0
    - The computation uses log-space for exponentiation to avoid overflow:
      λ^(i+1) = exp((i+1) · log(λ)) is more stable than direct pow()
    - When λ = 1, z_i = (1-1)/(1-1) → NaN, but eps in denominator
      prevents this: z_i ≈ (1 - 1) / eps ≈ 0, which is safe since
      λ = 1 means no decay (all weights = 1, so z_i = i+1, not 0).
      IMPORTANT: For λ = 1, the analytical formula is incorrect —
      z_i should equal i+1, not 0. The eps prevents NaN but does NOT
      give the correct normalization. This kernel should only be used
      with λ < 1. For λ = 1, a separate unweighted average kernel
      should be used.

    Memory Access Pattern
    --------------------
    - V tensor is accessed in 2D tiles of [BLOCK_N, D] — this is coalesced
      when the N dimension is contiguous (stride_vn = D for row-major layout)
    - Output is written in [BLOCK_M, D] tiles — coalesced under same assumption
    - lam is scalar per head (broadcast to all threads)
    - The phi matrix [BLOCK_M, BLOCK_N] is computed entirely in registers
      and never written to global memory

    Parameters
    ----------
    v_ptr : pointer
        Base pointer to V tensor (B, H, N, D).
    lam_ptr : pointer
        Base pointer to lambda tensor (H,).
    out_ptr : pointer
        Base pointer to output tensor (B, H, N, D).
    stride_vb, stride_vh, stride_vn, stride_vd : int
        Strides for V tensor dimensions.
    stride_ob, stride_oh, stride_on, stride_od : int
        Strides for output tensor dimensions.
    B, H, N, D : int
        Batch, heads, sequence length, head dimension.
    BLOCK_M : tl.constexpr
        Tile size for output rows (sequence dimension).
    BLOCK_N : tl.constexpr
        Tile size for V columns (sequence dimension).
    """
    # -------------------------------------------------------------------------
    # GRID GEOMETRY & BLOCK MAPPING
    # -------------------------------------------------------------------------
    # Each program instance processes one (batch, head) for one row tile.
    # pid_bh = b_idx * H + h_idx for 2D grid linearization.
    # pid_m identifies which chunk of BLOCK_M rows this instance computes.
    pid_bh = tl.program_id(0)  # Maps to Batch * Head instance
    pid_m = tl.program_id(1)   # Maps to the current row chunk (Sequence dimension)

    # De-linearize batch and head indices from the flattened BH dimension.
    # This avoids needing a 3D grid (B, H, M) which would be limited by
    # CUDA's max grid dimensions and add complexity.
    b_idx = pid_bh // H
    h_idx = pid_bh % H

    # Generate 1D offset vectors for rows (M) and hidden dimensions (D).
    # offs_m gives the absolute sequence positions for this tile.
    # offs_d gives the feature indices (0..D-1).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Compute base pointers for the current Batch-Head segment.
    # Adding the batch and head offsets once avoids doing it in the inner loop.
    v_ptrs = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    out_ptrs = out_ptr + b_idx * stride_ob + h_idx * stride_oh

    # -------------------------------------------------------------------------
    # MATHEMATICAL SETUP & PRE-COMPUTATIONS (Normalization Factor)
    # -------------------------------------------------------------------------
    # Load the scalar decay factor (λ) for the current head.
    # This is a single float shared by all threads in the block.
    lam = tl.load(lam_ptr + h_idx)
    
    # Accumulator for output rows: initialized to zero.
    # Using float32 for accumulation precision regardless of input dtype.
    # This is critical for numerical stability in long sequences where
    # many additions would accumulate error in float16/bfloat16.
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Small epsilon for numerical stability in division and log.
    # 1e-6 is a standard choice: small enough to not affect results
    # meaningfully, large enough to prevent division-by-zero.
    eps = 1e-6
    
    # Compute log(λ) for stable exponentiation. Adding 1e-9 to λ prevents
    # log(0) when λ = 0. While λ=0 is degenerate (only first position
    # contributes), it shouldn't crash the kernel.
    log_lam = tl.math.log(lam + 1e-9)
    
    # Pre-compute z_i analytically using geometric series sum:
    # z_i = Σ_{j=0}^{i} λ^j = (1 - λ^(i+1)) / (1 - λ)
    # This is computed for all i in this tile at once (vectorized).
    # WARNING: This formula is only correct for λ < 1.
    # For λ = 1, the correct z_i = i + 1 (arithmetic sum), but this
    # formula gives 0/0 → controlled by eps → incorrect result.
    lam_pow_i_plus_1 = tl.exp((offs_m + 1.0) * log_lam)
    z_i = (1.0 - lam_pow_i_plus_1) / (1.0 - lam + eps)

    # -------------------------------------------------------------------------
    # CAUSAL ATTENTION/DECAY LOOP
    # -------------------------------------------------------------------------
    # This loop implements the causal constraint by only iterating over
    # columns j up to the current row boundary (pid_m + 1) * BLOCK_M.
    # The loop bounds create a triangular iteration pattern where:
    # - pid_m=0 processes j in [0, BLOCK_M)     — 1 tile
    # - pid_m=1 processes j in [0, 2*BLOCK_M)   — 2 tiles
    # - pid_m=k processes j in [0, (k+1)*BLOCK_M) — k+1 tiles
    # This O(N²/BLOCK_N) scaling per program is the cost of causality.
    for j_start in range(0, (pid_m + 1) * BLOCK_N, BLOCK_N):
        # Column indices for this tile
        offs_n = j_start + tl.arange(0, BLOCK_N)
        
        # 2D Masking: Must satisfy ALL conditions:
        # 1. offs_n <= offs_m: Causality (j cannot attend to future i)
        # 2. offs_n < N: Column indices within sequence bounds
        # 3. offs_m < N: Row indices within sequence bounds
        # The broadcasting [None, :] vs [:, None] creates a 2D mask of
        # shape [BLOCK_M, BLOCK_N] where True means "valid to compute".
        mask = (offs_n[None, :] <= offs_m[:, None]) & (offs_n[None, :] < N) & (offs_m[:, None] < N)
        
        # Compute relative distances and exponential decay weights.
        # dist[i,j] = i - j (non-negative due to causal mask).
        # phi[i,j] = λ^(i-j) when mask is True, 0 otherwise.
        # Computing in log-space (exp(dist * log_lam)) is more numerically
        # stable than pow(lam, dist) for large distances.
        dist = offs_m[:, None] - offs_n[None, :]
        phi = tl.where(mask, tl.exp(dist * log_lam), 0.0)

        # Load a tile of V: shape [BLOCK_N, D]
        # The pointer arithmetic creates a 2D grid of pointers for batch
        # loading. out-of-bounds positions (j >= N) are filled with 0.0
        # via the 'other' parameter.
        v_block_ptrs = v_ptrs + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_block = tl.load(v_block_ptrs, mask=(offs_n[:, None] < N), other=0.0)
        
        # Accumulate: out_i += Σ_j λ^(i-j) · V[j]
        # This is a matrix multiplication: [BLOCK_M, BLOCK_N] × [BLOCK_N, D]
        # The decay weights phi are applied on-the-fly in the dot product,
        # avoiding the need to store a separate weighted V tensor.
        # Casting phi to match v_block dtype ensures efficient hardware
        # utilization (no mixed-precision dot product overhead).
        acc += tl.dot(phi.to(v_block.dtype), v_block)

    # -------------------------------------------------------------------------
    # EPILOGUE: WRITING BACK TO GLOBAL MEMORY
    # -------------------------------------------------------------------------
    # Apply normalization: out = acc / z_i
    # z_i[:, None] broadcasts from [BLOCK_M] to [BLOCK_M, D]
    # Adding eps to denominator prevents division by zero edge case.
    out = acc / (z_i[:, None] + eps)
    
    # Write output tile to global memory.
    # The mask ensures we don't write past the sequence length N.
    # Pointer arithmetic mirrors the V load pattern for consistency.
    out_block_ptrs = out_ptrs + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_block_ptrs, out, mask=(offs_m[:, None] < N))

def _cpu_global_decay(lam: "torch.Tensor", V: "torch.Tensor") -> "torch.Tensor":
    """
    Pure-PyTorch CPU fallback for triton_global_decay.
    Computes: out[i] = (sum_{j<=i} lam^(i-j) * V[j]) / z_i
    where z_i = (1 - lam^(i+1)) / (1 - lam)
    """
    import torch
    B, H, N, D = V.shape
    out   = torch.zeros_like(V)
    lam_b = lam.view(1, H, 1)          # (1, H, 1) for broadcast over D
    state = torch.zeros(B, H, D, device=V.device, dtype=V.dtype)
    for t in range(N):
        state = lam_b * state + V[:, :, t, :]          # (B, H, D)
        z     = (1.0 - lam_b ** (t + 1)) / (1.0 - lam_b + 1e-6)
        out[:, :, t, :] = state / z
    return out

def triton_global_decay(lam_scores, V):
    """
    Python wrapper for the flash_decay Triton kernel.

    Purpose
    -------
    Provides a clean Python interface that handles tensor validation,
    memory allocation, grid configuration, and kernel launch.

    Block Size Selection
    --------------------
    BLOCK_M = BLOCK_N = 32 is chosen because:
    - Fits well within typical GPU SRAM (32×32×4 bytes = 4KB for phi matrix,
      plus similar for V tile and accumulator)
    - Provides good occupancy: 32 threads per dimension maps efficiently
      to CUDA warp size (32 threads)
    - Powers of 2 avoid bank conflicts in shared memory
    - Small enough for fine-grained parallelism, large enough for good
      compute-to-memory ratio

    Grid Configuration
    ------------------
    - dim 0: B * H blocks (one per batch-head combination). These are
      completely independent and can execute in any order.
    - dim 1: ceil(N / BLOCK_M) blocks (tiled sequence length). These
      have dependencies only through global memory (not within kernel),
      so they also execute independently.

    Total parallelism: B * H * ceil(N/32) thread blocks.

    Assumptions & Constraints
    -------------------------
    - Input V must be contiguous in the last dimension (stride_vd = 1
      or D for row-major). Non-contiguous layouts may cause incorrect
      results or reduced performance.
    - λ values should be in (0, 1). λ = 1 is NOT handled correctly bya
      the analytical normalization (see kernel docstring).
    - Sequence length N should be reasonably large to amortize kernel
      launch overhead. For N < 64, a pure PyTorch implementation may
      be faster.

    Parameters
    ----------
    lam_scores : torch.Tensor
        Decay coefficients per head. Shape: (H,), dtype: float32.
        Values should be in (0, 1).
    V : torch.Tensor
        Value state tensor. Shape: (B, H, N, D), layout: row-major
        (last dim contiguous). Can be any floating dtype.

    Returns
    -------
    torch.Tensor
        Normalized exponentially decayed values. Shape: (B, H, N, D),
        same dtype as V.
    """
    if not V.is_cuda:
        return _cpu_global_decay(lam_scores, V)

    B, H, N, D = V.shape
    out = torch.empty_like(V)
    
    # Block sizes tuned for typical modern GPU SRAM/register constraints.
    # See docstring for selection rationale.
    BLOCK_M, BLOCK_N = 32, 32
    
    # Grid allocation: Parallelize across batches/heads (Dim 0) and
    # tiled sequence length (Dim 1). triton.cdiv rounds up for partial tiles.
    grid = (B * H, triton.cdiv(N, BLOCK_M))

    # Launch Triton Device Kernel
    # Strides are passed explicitly because the kernel uses pointer
    # arithmetic rather than tensor shapes. This allows the kernel to
    # work with any memory layout (though performance is best for
    # contiguous row-major tensors).
    flash_decay_fwd_kernel[grid](
        V, lam_scores, out,
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    return out