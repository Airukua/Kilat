---
name: assurance-chaos-engineering
description: Pre-Training Mathematical Assurance, Validation Gates, Active Fault Injection (Chaos/Saboteur), and Runtime Flight Telemetry for Kilat. Covers the 6-gate pre-flight verification pipeline, FP64 gradcheck, causal Jacobian checks, implementation equivalence, Wortsman/muP stability proxies, non-destructive fault injection, and anti-hallucination verification.
---

# Assurance & ML Chaos Engineering Skill

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Designing, implementing, or running pre-training validation gates (`kilat verify`, `src/kilat/assurance/`).
- Auditing mathematical correctness of attention, Global Decay, MLA, RoPE, RMSNorm, SwiGLU, or DeepSeekMoE.
- Proving equivalence between fast/kernelized paths (Triton, vectorized MoE, `HybridCache` incremental decode) and reference FP64 implementations.
- Writing or running active fault-injection / mutation tests (`kilat chaos`, `Saboteur` hooks).
- Implementing runtime health telemetry, silent data corruption (SDC) detection, or circuit-breaker callbacks (`FlightRecorderCallback`).
- Performing any refactor in `src/kilat/` where silent numerical regression is a risk.

> **Mandatory Cross-Reference**: Always read `docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md` and `docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md` alongside this skill.

---

## 2. Core Engineering Invariants

### 2.1 Zero-Trust Pre-Training Gate Principle
Never trust decreasing training loss alone as proof of correctness. Every architecture and engine configuration must pass **6 sequential pre-flight gates** before long-horizon GPU compute is authorized:
1. **Gate 1 — Theoretical & Init Sanity**: At step 0 with random initialization, pure cross-entropy loss must satisfy $|\mathcal{L}_{\text{CE}}(0) - \ln V| \le 0.15$, residual stream activation std must stay bounded across $L$ layers without exploding/vanishing, and 1D/norm/bias parameters must have `weight_decay == 0.0`.
2. **Gate 2 — Numerical & Causal Integrity**:
   - Analytical backward kernels/functions must match finite-difference `torch.autograd.gradcheck` in `torch.float64` within $10^{-6}$.
   - **Causal Future-Blindness**: $\max_{t' > t} \|\partial \mathbf{z}_{b,t} / \partial \mathbf{x}_{b,t'}\|_{\infty} == 0.0$.
   - **Batch Isolation**: Perturbing sample $b \neq 0$ in a batch must leave sample $0$ logits unchanged within floating-point roundoff ($\le 10^{-7}$ in FP64).
   - **Complete Gradient Connectivity**: 100% of `requires_grad=True` parameters must have `param.grad is not None`, `torch.isfinite(param.grad).all()`, and $\|\nabla \theta\| > 0$ on active paths (testing both CPU and CUDA paths when available).
3. **Gate 3 — Implementation Equivalence**:
   - Fast kernels vs FP64 pure-PyTorch reference: relative $L_\infty$ error $\le 10^{-5}$ (FP32), $\le 5 \times 10^{-3}$ (FP16), $\le 2 \times 10^{-2}$ (BF16) for both forward outputs and parameter gradients.
   - Full-sequence prefill logits at step $T-1$ must match incremental `HybridCache` decoding across `recall_ratio` $\in \{0.0, 0.5, 1.0\}$ and chunked prefill ($N > 1$ with cache).
4. **Gate 4 — Data Forensics**: All non-ignored labels $\in [0, V-1]$, every micro-batch has $\ge 1$ valid target token (`(labels[:, 1:] != -100).sum() > 0`), and 0 exact n-gram leaks between train and validation splits.
5. **Gate 5 — Micro-Dynamics & Stability Proxies**:
   - Single-batch memorization reaches $\mathcal{L}_{\text{CE}} < 0.05$ (dense) or $< 0.10$ (MoE) within 30–50 steps.
   - High-LR stress sweep ($3\times$ base LR, Wortsman et al. proxy) exhibits bounded max attention logits and zero NaN/Inf.
6. **Gate 6 — Determinism & DDP Parity**:
   - Checkpoint save at step $N$ + resume to $2N$ matches uninterrupted $1 \dots 2N$ **bitwise** ($\|\theta_{2N}^{\text{cont}} - \theta_{2N}^{\text{resumed}}\|_\infty == 0.0$).

### 2.2 Active Fault Injection (Testing the Tests)
A verification gate that has never been proven to fail on a known bug is untrusted.
- **Non-Destructive Sabotage Rule**: Fault injection (`Saboteur`) must **only** operate in memory at runtime via `contextlib.contextmanager`, `register_forward_hook`, `register_full_backward_hook`, or temporary tensor/state wrappers. **Never** mutate source files in `src/` on disk to test chaos scenarios.
- **Meta-Assertion Contract**: Every chaos scenario `C-01`..`C-08` passes **if and only if** the corresponding verifier gate or runtime circuit-breaker raises the expected structured `AssuranceError` / `CircuitBreakerTripped` code. If a sabotaged model passes `kilat verify`, the chaos test **FAILS**.

### 2.3 Zero-Sync Runtime Telemetry (`FlightRecorder`)
- **Rule**: Never insert `torch.isfinite(x).all()`, `.item()`, or `.any()` inside `nn.Module.forward()`. Device-to-host synchronization inside forward blocks CUDA streams and breaks `torch.compile(fullgraph=True)`.
- **Pattern**: Accumulate scalar health tensors asynchronously on device and materialize `.item()` **only** at `logging_steps` boundaries or when loss becomes non-finite at optimizer step boundaries.

---

## 3. Verification Protocol

Before signing off on any Assurance or Chaos implementation:
1. **Clean Baseline Pass**: Un-sabotaged model + synthetic debug config passes all 6 gates of `kilat verify` with 0 warnings.
2. **100% Saboteur Detection**: All 8 chaos scenarios (`C-01` through `C-08`) are caught with 0 false negatives and clean state restoration after exiting each `with saboteur.*:` context.
3. **Standard Suite Green**: `make check` and `make test` pass with $\ge 80\%$ coverage and 0 warnings.
