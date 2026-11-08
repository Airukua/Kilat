# GitHub Copilot Instructions — Kilat ML Training Engineering

This repository enforces strict ML Training Engineering standards. Read `AGENTS.md` and `.agents/skills/core/`.

## Key Guidelines
- In ML engineering, passing linting (`make check`) is not proof of training correctness.
- Code changes to architectures, loss functions, optimizers, schedulers, or data pipelines must be backed by numerical tests (`make test`).
- Enforce the step order: `optimizer.step()` before `scheduler.step()`.
- Enforce causal label shifts: `labels = input_ids[1:]`, loss masked on padding with `-100`.
- Enforce hardware-aware memory pinning: `pin_memory = torch.cuda.is_available()`.
- Enforce atomic checkpointing with full RNG state capture.
