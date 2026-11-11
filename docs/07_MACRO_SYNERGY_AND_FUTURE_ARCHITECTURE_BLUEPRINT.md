# 07 — Kilat Future Architecture Blueprint: Macro-Synergy & Interleaved Stacking

> **Doc ID**: `DOC-KILAT-07-MACRO-SYNERGY`  
> **Status**: Accepted Engineering Specification (Future Roadmap)  
> **Target Release**: Kilat Next-Gen Sub-Quadratic Hybrid LLM  
> **Preceding Reference**: [`02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md`](02_MATHEMATICAL_SPEC_AND_BUGFIX_PLAN.md), [`06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md`](06_MODEL_CERTIFICATION_AND_BENCHMARK_PLAN.md)

---

## 1. Executive Summary & Problem Diagnosis

In the standard Kilat-v2 architecture (`recall_ratio=0.5`), Global Decay and Multi-Head Latent Attention (MLA) are combined through an intra-layer split:

$$\mathbf{y}_t = W_O \begin{bmatrix} \mathbf{y}_t^{(g)} \\ \mathbf{y}_t^{(r)} \end{bmatrix} = W_O^{(g)} \mathbf{y}_t^{(g)} + W_O^{(r)} \mathbf{y}_t^{(r)}$$

### Critical Limitations of Static Concatenation
1. **Context-Blind Linear Superposition**: Matrices $W_O^{(g)}$ and $W_O^{(r)}$ are static. The model cannot dynamically suppress continuous-decay noise when a query token requires purely associative retrieval.
2. **Upstream-Representation Blindness**: The MLA query $\mathbf{q}_t$ is formed solely from the current token $\mathbf{x}_t$, without using the history accumulator $\mathbf{h}_t$ computed by Global Decay.
3. **Long-Sequence Capacity Bottleneck**: In MQAR evaluation ($N=256$, 16 key-value pairs), two MLA heads per layer interfered with one another, causing accuracy to collapse to **15.62%**.

---

## 2. Empirical Evidence from GPU Experiments

Direct tests were run on CUDA hardware with a micro-model ($d=64, H=4, L=2$):

### A. Accuracy & Associative Recall (MQAR)
- **Kilat-v2 Baseline (Concat 50/50)**:
  - $N=64$: 97.10% (1200 steps) / 67.19% (400 steps)
  - $N=128$: 100.00% (1000 steps) / 13.67% (400 steps)
  - $N=256$: **15.62%** (1000 steps) ❌ *Critical stagnation*
- **Macro Synergy (Interleaved L0 Decay $\to$ L1 MLA)**:
  - $N=64$: **98.80%** (1000 steps) / 74.22% (400 steps)
  - $N=128$: **100.00%** (1000 steps) / 33.20% (400 steps)
  - $N=256$: **96.09%** (1000 steps) 🏆 **(+80.47 percentage-point absolute improvement)**

### B. Compute & Memory Efficiency Profile
1. **KV Cache Footprint per Batch**:
   - $N=2048$: Ref Transformer = 1,024 KB | Kilat Concat = 320 KB | **Macro Synergy = 160 KB (6.4x vs Ref, 2.0x vs Base)**.
   - $N=8192$: Ref Transformer = 4,096 KB | Kilat Concat = 1,280 KB | **Macro Synergy = 640 KB**.
   - *Reason*: Layer 0 does not require a KV cache (0 bytes).
2. **Throughput Prefill**:
   - $N=512$: Macro Synergy = **423,808 tok/s** (+59% over Kilat Concat at 266,162 tok/s).
   - $N=1024$: Macro Synergy = **276,082 tok/s** (fastest, exceeding the Transformer at 269k tok/s).
3. **VRAM Reliability**:
   - The reference Transformer ran out of CUDA memory (OOM) at $N=4096$, while Macro Synergy and Kilat-v2 ran without OOM.

---

## 3. Interleaved-Stacking Architecture Specification

```
Layer 0 (Even)   : 100% Pure Global Decay Contextualizer
                   - 4-head Triton parallel prefix scan
                   - O(N) computation, KV cache = 0 bytes
                   - Output: mature causal temporal representation

Layer 1 (Odd)    : 100% Pure Latent MLA Associative Pointer
                   - 4-head latent attention + Decoupled RoPE
                   - Uses queries enriched with context from Layer 0
                   - Dedicates 100% of its capacity to precise memory retrieval
```

---

## 4. Code Migration & Compatibility Plan

For a future release:
1. Extend `KilatConfig` with the parameter `layer_layout: Literal["hybrid_50_50", "interleaved"]`.
2. Have `KilatTransformer` initialize blocks in alternating order when `layer_layout="interleaved"`.
3. Preserve the `LayerHybridCache` invariant with `latent_kv=None` on even-numbered layers.
