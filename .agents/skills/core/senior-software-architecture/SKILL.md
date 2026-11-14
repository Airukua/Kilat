---
name: senior-software-architecture
description: Senior Software Engineering Architecture Standards, Clean Layering, Modular Package Boundaries, Anti-Pattern Eradication, and Codebase Organization for Kilat ML Systems.
---

# Senior Software Architecture & Codebase Engineering Skill

## 1. When to Activate This Skill
Activate this skill whenever:
- Refactoring, modularizing, or reorganizing code in `src/kilat/`, `tests/`, `tools/`, or `configs/`.
- Adding new models, attention mechanisms, loss functions, or training loops.
- Identifying and eliminating code clutter ("kocar-kacir"), circular imports, or overlapping responsibilities.
- Designing CLI subcommands, model presets, or testing directory taxonomies.
- Reviewing PRs or code changes for software engineering elegance, scalability, and long-term maintainability.

> **Mandatory Cross-Reference**: Always read `docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md` and `docs/03_CODEBASE_RESTRUCTURE_PLAN.md` alongside this skill.

---

## 2. The 8 Architectural Pillars of Senior ML Engineering

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STRICT DOWNWARD DEPENDENCY LAYERING                  │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: CLI & Entrypoints           (src/kilat/cli.py)                │
│                            │                                           │
│ Layer 4: Workflows, Integrations     (assurance/, benchmarking/,       │
│          & Verification               integrations/)                   │
│                            │                                           │
│ Layer 3: Engine & Lifecycle          (engine/: trainer, checkpoint)    │
│                            │                                           │
│ Layer 2: Models & Data               (models/, data/)                  │
│                            │                                           │
│ Layer 1: Configs & Schemas           (configs/: typed configs)         │
│                            │                                           │
│ Layer 0: Leaf Utilities              (utils/: validators, formatting)  │
└────────────────────────────────────────────────────────────────────────┘
```

### Pillar 1: Strict Downward Layering (No Upward / Circular Dependencies)
Dependencies must flow strictly downward. Violating this creates hidden coupling and fragile code:
- **`models/` and `data/`** are pure domain representations. They must **NEVER** import `engine/`, `assurance/`, `benchmarking/`, or `cli.py`.
- **`engine/`** handles lifecycle execution (training loops, optimization, checkpointing). It imports `models/`, `data/`, and `configs/`, but must **NEVER** import `cli.py` or `benchmarking/`.
- **`assurance/` and `benchmarking/`** are evaluators that observe models and engines. They must **NEVER** be imported by `models/` or `engine/`.
- **`cli.py`** is the top-level composition root. It wires configs, models, and engines together.
- **Decoding primitives** (`inference/{generation_mixin,generation_config,logit_processor,sampler,beam_search,stopping_criteria}.py`) sit *below* `models/` (KilatTransformer mixes in `GenerationMixin`) and must never import `models/`, `engine/` or `data/`. The rest of `inference/` (`generator.py`, `export_hf.py`) sits above `models/`.
- **Enforcement**: `tests/integration/architecture/test_architecture_layering.py` AST-scans every `kilat.*` import (including function-local ones) against the allowed-layer matrix. Pre-existing violations live in `KNOWN_LAYERING_DEBT`, a ratchet: new entries fail the build, and fixed entries must be removed.
- **Utils re-export shims must be lazy** (PEP 562 `__getattr__`). An eager `from kilat.models...` in `kilat/utils/__init__.py` creates a circular import the moment `models/` uses any `utils/` leaf module.

### Pillar 2: Leaf Utilities Only in `utils/` (Zero Core Abstractions in `utils/`)
- **Anti-Pattern**: Placing core neural network abstractions (e.g. `BasePreTrainedModel`) or core loaders inside `src/kilat/utils/`.
- **Rule**: `src/kilat/utils/` is strictly for *domain-agnostic leaf utilities* (string formatters, ANSI banners, assertion validators).
- **Placement**:
  - Foundational model classes belong in `src/kilat/models/base.py`.
  - Checkpoint and model serializers belong in `src/kilat/engine/checkpoint.py` or `src/kilat/models/`.
  - Re-export shims can be placed in `utils/` only to maintain backward compatibility with external code.

### Pillar 3: Single-Door CLI (`kilat <subcommand>`)
- **Anti-Pattern**: Forcing developers to remember disparate script paths (e.g., `kilat verify` for gates, but `python tools/certify.py` for benchmarks).
- **Rule**: Every operational capability of Kilat must be accessible via `kilat <subcommand>`:
  - `kilat train` (launch training loop)
  - `kilat verify` (pre-flight assurance gates G1–G6)
  - `kilat chaos` (active fault injection C-01–C-08)
  - `kilat certify` (benchmarking zoology P1–P5)
  - `kilat generate` (text generation with HybridCache)
  - `kilat export-hf` (safetensors / HuggingFace exporter)
  - `kilat prepare-data` (zero-copy memmap preparation)
- `tools/` is strictly for optional offline developer scratch utilities, never for primary pipelines.

### Pillar 4: Single Unified Evidence Dossier
- **Anti-Pattern**: Scattering test results across `artifacts/verify/`, `reports/certification/`, and `verify_signoff/`.
- **Rule**: All assurance and certification reports write into **one canonical dossier directory** (`verify_signoff/` or `artifacts/dossier/`):
  - `verify_signoff/assurance_report.json`
  - `verify_signoff/chaos_report.json`
  - `verify_signoff/certification_report.json`
  - `verify_signoff/master_dossier.md` (combining all evidence according to SOP Bab 15).

### Pillar 5: 1:1 Mirroring Test Taxonomy
- **Anti-Pattern**: Splitting unit tests arbitrarily between `tests/unit/features/` and `tests/unit/pipelines/`.
- **Rule**: The test directory structure must mirror `src/kilat/` 1:1:
  ```
  tests/
  ├── unit/
  │   ├── models/        # mirrors src/kilat/models/
  │   ├── engine/        # mirrors src/kilat/engine/
  │   ├── data/          # mirrors src/kilat/data/
  │   ├── configs/       # mirrors src/kilat/configs/
  │   ├── inference/     # mirrors src/kilat/inference/
  │   └── assurance/     # mirrors src/kilat/assurance/
  ├── integration/       # multi-module pipelines & DDP
  └── shared/            # BDD fixtures, helpers
  ```

### Pillar 6: Decoupled Submodule Encapsulation
- **Anti-Pattern**: A single 600-line monolithic class handling two orthogonal mechanisms (e.g. `KilatAttention` doing both Global Decay and MLA with complex conditional branches).
- **Rule**: Encapsulate distinct computational primitives into focused modules:
  - `src/kilat/models/decay.py`: `GlobalDecayLayer` (pure $O(N)$ recurrence, Triton scan, zero KV cache).
  - `src/kilat/models/mla.py`: `LatentAttentionLayer` (pure MLA, compressed KV cache, decoupled RoPE).
  - `src/kilat/models/blocks.py`: `KilatBlock` orchestrating decay or MLA per layer based on `recall_pattern`.

### Pillar 7: Third-Party Ecosystem Integration Architecture (The Anti-Monolith & Zero-Duplication Standard)
- **Anti-Pattern**: *The 1000-Line Remote-Code Duplication Trap ("Sampah yang Susah Dirawat")*.
  - When bridging Kilat to external frameworks (Hugging Face, vLLM, PEFT, TRL), a junior tendency is to copy-paste the entire neural network (RMSNorm, SwiGLU, DeepSeekMoE, KilatAttention, blocks) into a single 1000+ line monolithic `modeling_<name>.py` file under `src/kilat/integrations/`.
  - **Why this is unmaintainable technical debt**:
    1. **Dual Maintenance & Code Drift**: Any architectural fix, numerical stability guard, or kernel improvement applied to `src/kilat/models/` will not propagate to the 1000-line copy. The two implementations inevitably diverge silently.
    2. **Violation of DRY & SSOT**: Repeating 800+ lines of identical tensor operations creates redundant clutter that bloats git history and complicates code audits.
    3. **Violation of SRP**: Normalization, FFN routing, attention kernels, caching, and causal wrappers are conflated into a single unreadable "God file".
- **The Senior Architecture Solution: Two-Tier Integration Pattern**:
  ```
  ┌────────────────────────────────────────────────────────────────────────┐
  │              TWO-TIER SENIOR INTEGRATION ARCHITECTURE                  │
  ├────────────────────────────────────────────────────────────────────────┤
  │ TIER 1: In-Tree Canonical Adapter (< 250 lines, Zero Duplication)      │
  │   src/kilat/integrations/<ecosystem>/                                  │
  │   ├── configuration_kilat.py  (~100 lines: PretrainedConfig bridge)    │
  │   ├── modeling_kilat.py       (~150 lines: Thin PreTrainedModel shim)  │
  │   └── __init__.py             (~30 lines: Explicit register hook)      │
  │   * Principle: Imports directly from `kilat.models.*` & `kilat.configs`│
  │   * Zero duplicate RMSNorm, zero duplicate MoE, zero duplicate kernels │
  │   * Strictly composes or subclasses canonical Kilat primitives         │
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

### Pillar 8: Module Complexity & File Line-Count Ceilings (< 400 Lines Budget)
- **Hard Ceiling**: **400 lines per file maximum**.
- **Soft Alert**: **300 lines per file**. When a module approaches 300 lines, investigate decomposing it into single-responsibility submodules.
- **Directory Nesting Budget**: Maximum depth of **3 levels** (`src/kilat/<domain>/<submodule>.py` or `src/kilat/integrations/<ecosystem>/<module>.py`). Arbitrarily deep directory hierarchies that obscure file ownership and create import mazes are forbidden.
- **Rule of Composition over Monolith**:
  - Thin translation adapters must ONLY adapt interfaces (e.g. converting `forward(input_ids, attention_mask, labels)` into Kilat's canonical tensor calls).
  - Adapters must never duplicate neural network layer math.
  - Parity tests (`tests/integration/integrations/`) must mathematically prove that native execution and adapter execution produce $\Delta = 0.0$ logit and loss divergence.

---

## 3. Canonical Preset Discovery Standard

Model presets (`kilat`, `kilat-macro-synergy`, `kilat-pure-decay`, `kilat-pure-attention`) must be accessible directly from the core models API, without requiring imports from `benchmarking`:

```python
# Canonical usage in training, inference, and research:
from kilat.models import KilatTransformer

# Fast factory instantiation:
model = KilatTransformer.from_preset("macro_synergy", vocab_size=50257)
```

---

## 4. Refactoring Protocol (Non-Breaking Migration)

When cleaning up codebase structure, always follow the **3-Step Safe Migration Protocol**:

```
Step 1: Move & Create Canonical Path
        Create the clean target class/file under the proper architectural directory.

Step 2: Backward-Compatibility Shim
        Leave a 1-line re-export with deprecation warning at the old path.
        (e.g., in src/kilat/utils/base_model.py: from kilat.models.base import BasePreTrainedModel)

Step 3: Update Internal Call Sites & Verify
        Update internal imports in src/ and tests/. Run `make test` to ensure 100% pass rate.
```

Never delete old import paths in a single breaking commit. Always preserve compatibility shims until the next major release.
