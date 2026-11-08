---
name: training-configuration
description: Training Configuration & Experiment Parameter Engineering for Kilat. Covers typed configuration dataclasses, YAML/JSON round-trip serialization, parameter divisibility constraints, hardware-aware VRAM heuristics, and CLI override hierarchies.
---

# Training Configuration Engineering Skill

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Modifying or extending configuration schemas (`MainConfig`, `KilatConfig`, `TrainingConfig`, `DataConfig`, `TokenizerConfig`).
- Reading or writing YAML / JSON experiment configuration files (`sample_config/*.yaml`).
- Adding CLI arguments or configuration override mechanisms.
- Adding hardware-aware configuration validation (VRAM estimation, head divisibility, context window limits).
- Ensuring configuration backward compatibility and migration.

---

## 2. Core Engineering Invariants

### 2.1 Configuration Precedence Hierarchy
Configuration parameters must resolve according to a deterministic hierarchy:
$$\text{CLI Explicit Flag} \succ \text{Experiment YAML File} \succ \text{Dataclass Defaults}$$
- Never hardcode training hyperparameters in source files (`trainer.py`, `train.py`). All parameters must flow through typed configs.

### 2.2 Serialization Round-Trip Invariant
Any configuration object must survive complete round-trip serialization without data loss or type mutation:
```python
original = MainConfig.from_yaml("configs/small.yaml")
exported_yaml = original.to_yaml()
restored = MainConfig.from_yaml_str(exported_yaml)
assert original == restored, "Configuration mutated across serialization round-trip"
```
- Non-serializable types (e.g. `torch.dtype`, lambdas, file descriptors) must be converted to strings (`"float32"`, `"bfloat16"`) prior to dumping.

### 2.3 Mathematical Divisibility & Hardware Constraints
All configurations must enforce structural architectural invariants before instantiation:
- **Head Dimension**: `dim % num_heads == 0`.
- **GQA / MLA Ratio**: `num_heads % num_kv_heads == 0` (or `num_heads % n_recall_heads == 0`).
- **Tensor Core Alignment**: `vocab_size` should ideally be divisible by 64 (or at least 8) to optimize CUDA warp execution efficiency.
- **MoE Routing**: `top_k <= num_experts`, `num_experts >= 1`.
- **Global Batch Size Contract**:
  $$\text{Global Batch Size} = \text{per\_device\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{num\_gpus}$$

### 2.4 Hardware-Aware Preflight Checks
Before launching training on constrained devices (e.g. RTX 4050 6GB / RTX 3060 12GB):
- Calculate estimated parameter memory:
  $$\text{Model VRAM (Bytes)} \approx \text{Parameters} \times \text{bytes\_per\_param}$$
  $$\text{Optimizer State (AdamW)} \approx \text{Parameters} \times 8 \text{ bytes (momentum + variance)}$$
- If estimated minimum VRAM > 90% of available GPU memory, the configuration validator must raise an informative warning suggesting activation checkpointing, BF16 precision, or smaller micro-batch size.

---

## 3. Verification Protocol

1. **Schema Validation**: Validate configs against edge cases (negative batch size, zero learning rate, missing dataset paths). Every invalid parameter must raise a specific `ValueError` with corrective guidance.
2. **Persistence Test**: Test export, reload, and verify identical parameter values in BDD format.
