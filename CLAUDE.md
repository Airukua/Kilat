# CLAUDE.md — Kilat ML Training Engineering Guidelines

Welcome to Kilat. When working as an AI agent in this codebase, adhere strictly to the engineering rules defined in `AGENTS.md` and domain skills in `.agents/skills/core/`.

## Essential Commands
- **Run quality check**: `make check` (executes `ruff check src tests` and `mypy src`)
- **Run test suite**: `make test` (executes BDD test suite with 80% coverage check and 0 warnings)
- **Run pre-commit**: `make pre-commit` (runs all git hooks across repository)
- **Format check**: `make format-check`

## Engineering Invariants
- **Silent Failure Prevention**: A training script running without error does not mean it is mathematically correct. Validate gradients, loss monotonicity, and state retention.
- **Optimization Order**: `optimizer.step()` must always precede `lr_scheduler.step()`.
- **Zero-Grad**: `zero_grad(set_to_none=True)` on gradient accumulation boundaries only.
- **Pin Memory**: Only set `pin_memory=True` if `torch.cuda.is_available()` is True.
- **Causal Masking**: Autoregressive targets must be shifted (`labels = input_ids[1:]`), pads masked with `-100`.
- **Checkpoints**: Atomic write pattern (write to temp file, rename); save full RNG states.

## Skills Map
- PyTorch Training: `.agents/skills/core/pytorch-training/SKILL.md`
- ML Data: `.agents/skills/core/ml-data-engineering/SKILL.md`
- Configuration: `.agents/skills/core/training-configuration/SKILL.md`
- Checkpointing: `.agents/skills/core/checkpoint-reproducibility/SKILL.md`
- ML Testing: `.agents/skills/core/ml-testing/SKILL.md`
- GPU Performance: `.agents/skills/core/gpu-performance/SKILL.md`
