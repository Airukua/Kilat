---
name: ml-testing
description: ML Testing & Invariant Verification Engineering for Kilat. Covers tensor shape invariants, gradient verification, overfit-one-batch smoke tests, determinism checks, BDD formatting standards, and zero-warning test suites.
---

# ML Testing & Verification Skill

> **Mandatory Spec Reference**: Read `docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md` and `docs/04_ASSURANCE_AND_CHAOS_ENGINEERING_PLAN.md` before writing tests. Note that CPU-only unit tests execute `_cpu_global_decay` rather than CUDA `@triton.jit` kernels, and `KilatTransformer.forward` shifts causal labels internally (`logits[:, :-1]`, `labels[:, 1:]`).

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Writing, updating, or debugging tests in `tests/unit/` or `tests/integration/`.
- Verifying model architecture, training loop, or data pipeline correctness.
- Checking test coverage (enforcing $\ge 80\%$ statement coverage).
- Eliminating test warnings and maintaining clean terminal execution.
- Implementing BDD test scenarios following the `Given / When / Then / And` pattern.

---

## 2. Core Engineering Invariants

### 2.1 The 4 Pillars of ML Testing
In Kilat, unit and integration tests must test mathematical and operational contracts, not trivial mock returns:
1. **Tensor Shape Invariants**: Assert exact multidimensional tensor shapes through every layer transformation (e.g. $[B, S, D] \to [B, S, V]$).
2. **Gradient Existence & Flow**: Assert that `param.grad` exists, is non-zero, and is free of NaNs/Infs for all trainable weights after `.backward()`.
3. **Loss Monotonicity (Overfit One Batch)**: A small model running on a single micro-batch must significantly decrease loss within 20 iterations:
   $$\mathcal{L}_{20} < 0.5 \times \mathcal{L}_{0}$$
4. **Determinism & Seeds**: Two forward passes with the same seed and input must produce bitwise identical outputs:
   $$\| f(x; \theta)_{\text{run 1}} - f(x; \theta)_{\text{run 2}} \|_{\infty} == 0.0$$

### 2.2 BDD Terminal Formatting Contract
Tests must follow the structured BDD layout using `tests.shared.bdd`:
```text
PASS/FAIL  <test-level>  <test-file-path>
<Scenario Title>
  given <precondition>
    when <action>
      ✓ <expected behavior> (<duration> ms)
```
- Standard pytest dots (`.`) and percentage progress bars must be suppressed via `addopts = "-s -q"` and `conftest.py` hooks.
- Test sessions must begin with the clean ASCII KILAT banner.

### 2.3 Zero-Warning Standard
- All PyTorch warnings (e.g. `pin_memory` without CUDA, `lr_scheduler.step()` before `optimizer.step()`, tokenizer vocab resolution warnings) must be eliminated through correct engineering or targeted `warnings.catch_warnings()`.
- Test suites must complete with **0 warnings**.

---

## 3. Verification Protocol

Run the full quality gate:
```bash
make check   # ruff + mypy (0 issues)
make test    # BDD tests + >= 80% coverage + 0 warnings
```
Never commit code where `make check` or `make test` fails.
