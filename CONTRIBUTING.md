# Contributing to Kilat

Thank you for your interest in contributing to **Kilat**! ⚡

Kilat is a modern, high-performance hybrid language modeling framework that combines **Sub-Quadratic Global Decay ($O(1)$ recurrent scaling)**, **DeepSeekMoE (Sparse Mixture-of-Experts)**, and **Latent Multi-Head Attention (MLA)**.

We welcome contributions from researchers, ML engineers, kernel writers, and open-source enthusiasts. To maintain world-class training stability and mathematical reproducibility, we adhere to strict engineering standards.

---

## Table of Contents

- [The 5 Golden Invariants](#the-5-golden-invariants)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
- [Development Workflow](#development-workflow)
  - [Branching Strategy](#branching-strategy)
  - [Conventional Commits](#conventional-commits)
- [Code Quality & Standards](#code-quality--standards)
  - [Linting and Formatting (`ruff`)](#linting-and-formatting-ruff)
  - [Type Checking (`mypy`)](#type-checking-mypy)
  - [Sonar Quality & Hygiene Auditor](#sonar-quality--hygiene-auditor)
  - [Gate 0 Security & Supply Chain](#gate-0-security--supply-chain)
- [Testing & Pre-Flight Assurance](#testing--pre-flight-assurance)
  - [Running Unit & Integration Tests](#running-unit--integration-tests)
  - [Pre-Flight Verification (`kilat verify`)](#pre-flight-verification-kilat-verify)
  - [Active Chaos Engineering (`kilat chaos`)](#active-chaos-engineering-kilat-chaos)
- [Domain-Specific Contribution Guidelines](#domain-specific-contribution-guidelines)
  - [Architectures & Layers (`src/kilat/models/`)](#architectures--layers-srckilatmodels)
  - [Training Loop & Optimizers (`src/kilat/engine/`)](#training-loop--optimizers-srckilatengine)
  - [Ecosystem & HF Integrations (`src/kilat/integrations/`)](#ecosystem--hf-integrations-srckilatintegrations)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Code of Conduct](#code-of-conduct)

---

## The 5 Golden Invariants

In ML systems, bugs rarely raise loud exceptions; instead, they silently corrupt gradients, leak attention across sequence boundaries, skip LR warmups, or produce un-resumable checkpoints. Every contribution must respect Kilat's 5 Golden Invariants:

1. **Numerical Proof over Syntax Passing**: Linters (`ruff`) and type-checkers (`mypy`) are baseline prerequisites, not proof of completion. Any architectural, loss, or optimization change must include numerical assertions and invariant tests.
2. **Gradient & Optimization Sanity**:
   - `optimizer.step()` must always precede `lr_scheduler.step()` (or scaler unscaling order).
   - Zero-grad timing must be strictly preserved during gradient accumulation.
   - Every trainable parameter must receive non-zero, finite gradients during backprop; frozen parameters must have zero or `None` gradient.
3. **Data Leakage & Causal Integrity**:
   - Autoregressive targets must be strictly causal (`labels = input_ids[1:]`, loss masked on prompt/padding with `-100`).
   - Sample boundaries must never leak attention across concatenated sequences without document-boundary masking.
4. **Deterministic Checkpointing & State Recovery**:
   - Resuming training from step $N$ must match uninterrupted training state exactly.
   - Checkpoints must preserve model weights, optimizer states, LR scheduler, AMP scaler, step counters, and PyTorch/CUDA/NumPy/Python RNG states.
   - File saves must be atomic (written to a temporary buffer/path, then renamed) to prevent corrupt files upon preemption.
5. **Hardware & Acceleration Awareness**:
   - Dataloaders must only enable `pin_memory=True` when hardware accelerators are present (`torch.cuda.is_available()`).
   - Mixed precision (`torch.amp.autocast`) must respect target hardware capabilities (BF16 for Ampere+, FP16 with GradScaler for older architectures).

---

## Getting Started

### Prerequisites

- **Python**: $\ge 3.10$ (Python 3.11 or 3.12 recommended)
- **PyTorch**: $\ge 2.0.0$ (with CUDA support recommended for GPU training)
- **Git** & **Make**

### Environment Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Kilat.git
   cd Kilat
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using conda
   conda create -n kilat-dev python=3.12 -y
   conda activate kilat-dev

   # Or using venv
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Kilat with development and integration dependencies**:
   ```bash
   make install-dev
   # or manually:
   pip install -e ".[dev,integrations,reporting]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   make pre-commit-install
   ```

---

## Development Workflow

### Branching Strategy

- Always branch off `main`.
- Use descriptive branch names:
  - `feat/add-new-attention-kernel`
  - `fix/moe-aux-loss-reduction`
  - `docs/update-generation-guide`
  - `chaos/add-kv-cache-drift-test`

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature or capability (e.g., `feat: implement flash linear attention kernel`)
- `fix:` A bug fix (e.g., `fix: prevent dynamic cache index error in HF generate`)
- `docs:` Documentation changes (e.g., `docs: add contributing guide`)
- `test:` Adding or updating tests (e.g., `test: add causal mask isolation unit test`)
- `refactor:` Code restructuring without changing external behavior
- `perf:` Performance optimizations (e.g., `perf: optimize Triton decay scan`)
- `chore:` Build scripts, dependencies, or maintenance

---

## Code Quality & Standards

Run all quality checks locally before opening a pull request:

```bash
make check
```

This runs linting (`ruff`), type checking (`mypy`), and the Sonar code quality auditor.

### Linting and Formatting (`ruff`)

Kilat uses `ruff` for ultra-fast linting and formatting.

```bash
# Check for lint issues
make lint

# Automatically fix fixable issues
make lint-fix

# Format code
make format
```

### Type Checking (`mypy`)

All new code in `src/` must be strictly typed.

```bash
make type-check
```

### Sonar Quality & Hygiene Auditor

We enforce clean-code hygiene:
- No hardcoded password/token constants without `# noqa: S105`.
- Avoid duplicating string literals $\ge 3$ times; define module-level constants.
- Avoid raw `assert` statements inside `src/` (use explicit custom exceptions; `assert` is reserved for tests).

```bash
make quality
```

### Gate 0 Security & Supply Chain

Kilat features an automated Gate 0 security scanner based on OWASP LLM/ML Top 10 and NCSC guidelines:

```bash
make security
```

---

## Testing & Pre-Flight Assurance

### Running Unit & Integration Tests

Kilat has a modular test suite under `tests/`:

```bash
# Run all tests
make test

# Run a specific test module
pytest tests/unit/models/ -v
pytest tests/integration/assurance/test_hf_chaos.py -v
```

### Pre-Flight Verification (`kilat verify`)

Before any major model training run or architectural PR, run Kilat's 6-Gate verification harness:

```bash
make verify
```

This checks:
- **Gate 0**: Security & supply chain AST audit
- **Gate 1**: Structural configuration validity
- **Gate 2**: Numerical & dtype stability
- **Gate 3**: Causal integrity & data leak absence
- **Gate 4**: Loss & gradient convergence
- **Gate 5**: Checkpoint determinism & RNG parity
- **Gate 6**: Hardware & throughput telemetry

### Active Chaos Engineering (`kilat chaos`)

Kilat includes an active **Saboteur** suite to test resilience against adversarial training and deployment faults:

```bash
# Run all standard training chaos scenarios (C-01 .. C-08)
make chaos

# Run Hugging Face integration chaos scenarios (HFC-01 .. HFC-08)
kilat chaos -s hf
```

---

## Domain-Specific Contribution Guidelines

### Architectures & Layers (`src/kilat/models/`)
- Ensure all attention/recurrent modules uphold causal masking ($O(1)$ linear or hybrid).
- Do not introduce quadratic memory dependencies where sub-quadratic claims are made.
- Implement both training mode (`forward` with full parallel scan/causal mask) and autoregressive inference mode with state/cache tracking.

### Training Loop & Optimizers (`src/kilat/engine/`)
- Follow the canonical optimization order:
  1. `loss.backward()` (or `scaler.scale(loss).backward()`)
  2. Unscale gradients
  3. `torch.nn.utils.clip_grad_norm_`
  4. `optimizer.step()` (or `scaler.step(optimizer)`)
  5. `scaler.update()`
  6. `lr_scheduler.step()`
  7. `optimizer.zero_grad(set_to_none=True)` (at accumulation boundaries)

### Ecosystem & HF Integrations (`src/kilat/integrations/`)
- Remote code export files (`modeling_kilat.py`, `configuration_kilat.py`) must be self-contained and independently importable by vanilla `transformers`.
- Respect `_supports_default_dynamic_cache(cls) -> False` when custom recurrent states are used.
- Ensure exported checkpoints load cleanly via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`.

---

## Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feat/your-feature-name
   ```
2. **Open a Pull Request** against the `main` branch on GitHub.
3. **Fill out the PR Template**:
   - What problem does this PR solve?
   - Which invariant(s) does this affect?
   - How was it tested? (Attach `make check`, `make test`, or `kilat verify` output).
4. **Address Review Feedback**: Community maintainers will review your PR. Be open to feedback and iterate on improvements.

---

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

---

*Thank you for helping make Kilat the fastest, most reliable open-source foundation model framework!*
