---
name: integration-architecture
description: Senior Software Architecture Standard for Third-Party Framework Integrations (Hugging Face, vLLM, PEFT, TRL) — Preventing Monolithic Duplication Traps, Enforcing Thin Adapters, and Synthesizing Standalone Distribution Artifacts.
---

# Senior Integration Architecture & Ecosystem Interoperability Skill

## 1. When to Activate This Skill
Activate this skill whenever:
- Designing or maintaining integration bridges to external ML ecosystems (**Hugging Face Hub**, **vLLM**, **PEFT LoRA**, **TRL SFT/DPO**, **DeepSpeed**, or **Megatron**).
- Exporting models with custom architectures using `auto_map` and `trust_remote_code=True`.
- Refactoring large, messy, or duplicate files in `src/kilat/integrations/`.
- Reviewing code for modularity, line-count ceilings, and clean separation of concerns.

> **Mandatory Rule**: In ML frameworks, never maintain two copies of the same tensor operations. An integration adapter must NEVER duplicate mathematical definitions from `src/kilat/models/`.

---

## 2. The Core Anti-Pattern: The 1000-Line Duplication Trap ("Sampah yang Susah Dirawat")

### The Root Cause: The Remote-Code Fallacy
Frontier registries like Hugging Face Hub support arbitrary custom architectures via `auto_map` metadata in `config.json` and `trust_remote_code=True`.
For end-users to load a model without having `pip install kilat` on their system, the model files on the Hub must not fail with `ModuleNotFoundError: No module named 'kilat'`.

A junior engineer often responds to this requirement by:
1. Copy-pasting RMSNorm, SwiGLU, DeepSeekMoE, Attention, Triton kernels, and Blocks from `src/kilat/models/` into a single monolithic `src/kilat/integrations/huggingface/modeling_kilat.py`.
2. Checking this 1000+ line "God file" directly into the source control repository under `src/`.

### Why This Is Unmaintainable Technical Debt ("Sampah")
1. **Silent Code Drift**:
   When bug fixes, numerical stability guards (e.g. $\epsilon = 10^{-6}$, FP16 square-overflow prevention), or kernel speedups are committed to `src/kilat/models/`, developers forget to mirror them into the 1000-line integration file. The two implementations silently diverge.
2. **Gross Violation of DRY & Single Source of Truth (SSOT)**:
   Over 80% of the monolithic adapter is pure redundancy. Reviewers and maintainers must audit identical logic twice.
3. **Violation of Single Responsibility Principle (SRP)**:
   A single file handles normalization, feed-forward routing, attention decay, block sequencing, generation mixins, and Hugging Face causal LM wrappers.
4. **Git History & Review Friction**:
   A 1000-line monolith generates massive diffs for trivial changes and obscures architectural intent.

---

## 3. The Senior Solution: The Two-Tier Integration Pattern

Senior ML software engineers strictly separate **In-Tree Development Architecture** from **Out-of-Tree Distribution Packaging**:

```
┌────────────────────────────────────────────────────────────────────────┐
│              TWO-TIER SENIOR INTEGRATION ARCHITECTURE                  │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 1: In-Tree Canonical Adapter (< 200 lines, Zero Duplication)      │
│   src/kilat/integrations/<ecosystem>/                                  │
│   ├── configuration_kilat.py  (~100 lines: PretrainedConfig bridge)    │
│   ├── modeling_kilat.py       (~100 lines: Thin PreTrainedModel shim)  │
│   └── __init__.py             (~30 lines: Explicit register hook)      │
│   * Principle: Imports directly from `kilat.models.*` & `kilat.configs`│
│   * Zero duplicate RMSNorm, zero duplicate MoE, zero duplicate kernels │
│   * Strictly composes or subclasses canonical Kilat primitives         │
│   * Purpose: Native registration & in-process HF ecosystem tools       │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 2: Out-of-Tree Synthesized Hub Bundle (Build/Export Time Only)    │
│   src/kilat/inference/export_hf.py (Automated Code Bundler)            │
│   * Principle: Single Source of Truth (SSOT).                          │
│   * When `kilat export-hf` executes:                                   │
│     The exporter automatically synthesizes/inlines the self-contained  │
│     remote-code file into the target checkpoint directory for Hub      │
│     upload (`trust_remote_code=True`).                                 │
│   * The 1000-line standalone file is an EXPORT ARTIFACT,               │
│     NEVER tracked as hand-edited source code in `src/`!                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Architectural Rules & Governance

### Rule 1: File Line-Count Budget
- **Hard Ceiling**: **400 lines per file maximum**.
- **Soft Alert**: **300 lines per file**. Modules approaching 300 lines must be reviewed for decomposition.
- Any integration file exceeding 400 lines will be rejected in architecture review.

### Rule 2: Package Nesting Budget
- **Maximum Directory Depth**: **3 levels** (`src/kilat/integrations/<ecosystem>/<module>.py`).
- Nested folder hierarchies deeper than 3 levels are prohibited to prevent import mazes.

### Rule 3: Thin Translation Shims
- Adapters must ONLY translate signatures:
  - Converting Hugging Face's `forward(input_ids, past_key_values, labels, ...)` to Kilat's native calls.
  - Exposing `get_input_embeddings()`, `set_input_embeddings()`, and `prepare_inputs_for_generation()`.
- Adapters must NEVER re-implement layers or kernels.

### Rule 4: Clean Opt-In Registration
- Never execute global registrations at root `__init__.py` import time.
- Provide clean, explicit hooks:
  ```python
  from kilat.integrations.huggingface import register_kilat
  register_kilat()
  ```

### Rule 5: Bidirectional Parity Guarantees
- Every integration adapter must be verified by automated numerical parity tests:
  - $\max | \text{logits}_{\text{native}} - \text{logits}_{\text{adapter}} | < 10^{-5}$ (target: $0.0$).
  - $| \text{loss}_{\text{native}} - \text{loss}_{\text{adapter}} | < 10^{-5}$ (target: $0.0$).
  - Identical greedy decoding sequences across $K$ autoregressive steps.
