# AGENTS.md — Kilat AI Coding Agent Engineering Standard

> **Core Philosophy**: Kilat requires an AI coding agent that deeply understands **ML Training Engineering**, not just Python syntax or PyTorch mechanics.
> In ML systems, errors rarely raise explicit exceptions. A model can run smoothly and report decreasing loss while silently corrupting gradients, learning on leaky data, breaking RNG determinism, skipping LR scheduler updates, or saving truncated checkpoints.

---

## 1. The 5 Golden Invariants of Kilat

Every AI coding agent operating in this repository must uphold these invariants:

1. **Numerical Proof over Syntax Passing**:
   Passing linters (`ruff`) and type-checkers (`mypy`) is a prerequisite, not proof of completion. Any modification to architectures, datasets, optimizers, or training loops must be proven via numerical assertions and invariant tests.
2. **Gradient & Optimization Sanity**:
   - `optimizer.step()` must always precede `lr_scheduler.step()` (or scaler unscaling order).
   - Zero-grad timing must be strictly preserved during gradient accumulation.
   - Every trainable parameter must receive non-zero, finite gradients during backprop; frozen parameters must have zero or `None` gradient.
3. **Data Leakage & Causal Integrity**:
   - Autoregressive targets must be strictly causal (`labels = input_ids[1:]`, loss masked on prompt/padding with `-100`).
   - Sample boundaries must never leak attention across concatenated sequences without document-boundary masking.
4. **Deterministic Checkpointing & State Recovery**:
   - Resuming training from step $N$ must match uninterrupted training state exactly.
   - Checkpoint state must preserve: model weights, optimizer states, lr scheduler, AMP scaler, step counters, and PyTorch/CUDA/NumPy/Python RNG states.
   - Saves must be atomic (written to temporary buffer/path, then renamed) to prevent corrupt files upon preemption.
5. **Hardware & Acceleration Awareness**:
   - Dataloaders must only enable `pin_memory=True` when hardware accelerators are present (`torch.cuda.is_available()`).
   - Mixed precision (`torch.cuda.amp.autocast`) must respect target hardware capabilities (BF16 for Ampere+, FP16 with GradScaler for older architectures).

---

## 2. Modular Skill Architecture

Specialized domain knowledge is organized under `.agents/skills/`. Agents must invoke and follow the relevant skill specification when working within that domain:

```
.agents/
└── skills/
    ├── core/
    │   ├── pytorch-training/          # Autograd, loops, optimizers, schedulers, AMP, stability
    │   ├── ml-data-engineering/       # Tokenization, sequence packing, causal masking, leak prevention
    │   ├── training-configuration/    # Typed configs, YAML/JSON validation, hardware heuristics
    │   ├── checkpoint-reproducibility/ # Atomic saves, full RNG restore, resumption parity
    │   ├── ml-testing/                # Invariant tests, overfit-one-batch, gradient checks, BDD
    │   └── gpu-performance/           # VRAM profiling, activation checkpointing, throughput
    ├── research/                      # Specialized architectures (MoE, MLA, Triton kernels, DDP)
    └── product/                       # DX, documentation, release engineering
```

---

## 3. Agent Execution Protocol (The 5-Step Workflow)

When given any engineering task, the agent must execute these steps sequentially:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Reconnaissance        → Read existing configs, models, tests        │
│ 2. Invariant Analysis    → Identify tensor shapes, gradients, RNGs     │
│ 3. Implementation        → Apply modular changes with safety guards    │
│ 4. Numerical Proof       → Overfit one batch & run BDD verification    │
│ 5. Verification Artifact → Report metrics, memory delta, and test log │
└────────────────────────────────────────────────────────────────────────┘
```

Never declare a task finished without showing verifiable test output from `make test` or dedicated verification scripts.

---

## 4. Multi-Harness Compatibility

This repository supports all major AI coding harnesses:
- **Cursor IDE**: Read [`.cursorrules`](file:///.cursorrules) and [`.cursor/rules/`](file:///.cursor/rules/).
- **Claude Code**: Read [`CLAUDE.md`](file:///CLAUDE.md) (mirrors `AGENTS.md`).
- **Windsurf**: Read [`.windsurfrules`](file:///.windsurfrules).
- **GitHub Copilot**: Read [`.github/copilot-instructions.md`](file:///.github/copilot-instructions.md).
- **Antigravity / Gemini CLI**: Uses `AGENTS.md` and `.agents/skills/` natively.
