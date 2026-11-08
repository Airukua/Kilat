# GitHub Copilot Instructions — Kilat ML Training Engineering

This repository enforces strict ML Training Engineering standards. Read `AGENTS.md` and `.agents/skills/core/`.

## Key Guidelines
- In ML engineering, passing linting (`make check`) is not proof of training correctness.
- Code changes to architectures, loss functions, optimizers, schedulers, or data pipelines must be backed by numerical tests (`make test`).
- Enforce the step order: `optimizer.step()` before `scheduler.step()`.
- Enforce causal label shifts: `labels = input_ids[1:]`, loss masked on padding with `-100`.
- Enforce hardware-aware memory pinning: `pin_memory = torch.cuda.is_available()`.
- Enforce atomic checkpointing with full RNG state capture.
- Always consult `docs/01_AGENT_ANTI_HALLUCINATION_CONTRACT.md` through `docs/05_STEP_BY_STEP_EXECUTION_CHECKLIST.md` and `.agents/skills/core/assurance-chaos-engineering/SKILL.md` before modifying `src/` or `tests/`.
