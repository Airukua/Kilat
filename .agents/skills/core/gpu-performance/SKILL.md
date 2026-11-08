---
name: gpu-performance
description: GPU Memory & Throughput Performance Engineering for Kilat. Covers CUDA memory profiling, activation checkpointing, DataLoader bottleneck diagnosis, mixed precision optimization, and throughput measurement (tokens/sec, MFU).
---

# GPU Performance Engineering Skill

> **Mandatory Spec Reference**: Consult `docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md` and `docs/02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md` (specifically M-01 for CUDA `@triton.jit` autograd wrapping via `torch.autograd.Function` and M-07 for removing synchronous `.all()`/`.any()`/`.item()` CUDA stream stalls from `nn.Module.forward()`).

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Diagnosing CUDA Out of Memory (OOM) errors.
- Measuring or optimizing training throughput (tokens/second, TFLOPS, MFU).
- Enabling or tuning activation checkpointing (gradient checkpointing).
- Benchmarking Triton kernels or custom attention mechanisms.
- Profiling memory footprints (model weights, optimizer states, KV cache, activations).
- Investigating low GPU utilization or DataLoader starvation.

---

## 2. Core Engineering Invariants

### 2.1 The VRAM Breakdown Model
Before attempting performance tuning, decompose GPU memory usage:
$$\text{Total VRAM} = \text{Model Parameters} + \text{Optimizer States} + \text{Gradients} + \text{Activations} + \text{Workspace/Cache}$$

| Component | FP32 | FP16/BF16 | AdamW Optimizer States |
| :--- | :--- | :--- | :--- |
| **Model Weights** | $4 \times \Phi$ bytes | $2 \times \Phi$ bytes | - |
| **Gradients** | $4 \times \Phi$ bytes | $2 \times \Phi$ bytes | - |
| **Optimizer States** | - | - | $8 \times \Phi$ bytes (FP32 master copies + momentum) |
| **Total Static State** | $8 \times \Phi$ bytes | $4 \times \Phi$ bytes | $+ 8 \times \Phi$ bytes $= 12 \times \Phi$ bytes |

For a 100M parameter model ($\Phi = 10^8$):
- Static training state in 16-bit mixed precision: $\approx 1.2\text{ GB}$.
- Remaining VRAM on an 8GB or 12GB GPU is allocated to dynamic activations and batch size.

### 2.2 Activation Checkpointing Strategy
- **When to Use**: When sequence length $S \ge 2048$ or batch size cannot fit due to activation memory quadratic scaling.
- **Trade-off**: Drops forward activation tensors and recomputes them during `.backward()`. Saves $50\text{--}70\%$ of activation VRAM at the cost of $\approx 25\text{--}30\%$ extra forward compute.
- **Kilat Implementation**:
  ```python
  if model.is_gradient_checkpointing:
      hidden_states = torch.utils.checkpoint.checkpoint(
          layer_block, hidden_states, use_reentrant=False
      )
  ```
  *Always use `use_reentrant=False` in modern PyTorch.*

### 2.3 Throughput & MFU Measurement Standards
- **Tokens per Second**:
  $$\text{Throughput} = \frac{\text{Micro-Batch Size} \times \text{Sequence Length} \times \text{Accumulation Steps}}{\Delta t_{\text{step}}}$$
- **Model FLOPs Utilization (MFU)**: Measure hardware compute efficiency relative to peak theoretical TFLOPS (e.g. RTX 4090: 82.58 FP16/BF16 TFLOPS without sparsity).
- **DataLoader Starvation Check**:
  Measure the time spent in `next(data_iterator)` vs `model(input_ids)`. If data fetch time exceeds 15% of step time, increase `num_workers`, use `persistent_workers=True`, or convert datasets to memmap.

---

## 3. Verification Protocol

1. **Peak Memory Audit**:
   ```python
   torch.cuda.reset_peak_memory_stats()
   # Execute forward + backward + step
   peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
   print(f"Peak VRAM: {peak_mb:.1f} MB")
   ```
2. **Deterministic Speedup Proof**: Any proposed optimization must be proven via reproducible benchmark scripts recording tokens/second before and after the change. Never claim speedups based on theory alone.
