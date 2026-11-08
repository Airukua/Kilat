---
name: checkpoint-reproducibility
description: Training Recovery, Checkpoint Integrity, and Experiment Reproducibility Engineering for Kilat. Covers atomic saving, state_dict capture, full RNG restoration, seamless resumption parity, and preemption safety.
---

# Checkpointing & Reproducibility Skill

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Saving or restoring model checkpoints (`save_checkpoint`, `load_checkpoint`, `KilatTrainer`).
- Resuming interrupted training runs from disk.
- Managing RNG seeds and determinism (`torch.manual_seed`, `torch.cuda.manual_seed_all`).
- Pruning old checkpoints or managing `save_total_limit`.
- Ensuring safe model export (`safetensors`, HuggingFace format conversion).

---

## 2. Core Engineering Invariants

### 2.1 The Complete State Checkpoint Contract
A training checkpoint is **not** just model weights. To be fully reproducible, a checkpoint dictionary must capture:
```python
checkpoint = {
    # 1. Model State
    "model_state_dict": unwrap_model(model).state_dict(),
    # 2. Optimization State
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict() if scaler else None,
    # 3. Step & Metric Metadata
    "global_step": step,
    "epoch": epoch,
    "best_metric": best_metric,
    "config": config.to_dict(),
    # 4. Complete RNG State
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    },
}
```

### 2.2 Atomic Saving to Prevent Corruptions
- **Preemption Safety**: Never write directly to the target checkpoint filename. If training is preempted or interrupted mid-write, the checkpoint will be left in a corrupted, half-written state.
- **Atomic Pattern**:
  ```python
  temp_path = f"{target_path}.tmp.{os.getpid()}"
  torch.save(checkpoint, temp_path)
  os.replace(temp_path, target_path)  # Atomic on POSIX filesystems
  ```

### 2.3 Resumption Parity Verification
- **Invariant**: If training runs uninterrupted for 100 steps, vs running for 50 steps, saving checkpoint, and resuming from step 50 to 100:
  $$\text{Loss}_{\text{uninterrupted}}(t) \equiv \text{Loss}_{\text{resumed}}(t) \quad \forall t \in [51, 100]$$
- If losses diverge upon resumption, one or more of the following is broken:
  1. Learning rate scheduler was re-initialized without `load_state_dict`.
  2. Optimizer momentum / variance buffers were dropped.
  3. RNG states were reset to default instead of restored.
  4. DataLoader position or dataset sampler seed was not synchronized.

### 2.4 Device & Architecture Independence
- When reloading checkpoints on different devices (e.g. trained on CUDA, inspected on CPU):
  ```python
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  ```
- Strip any `_orig_mod.` (from `torch.compile`) or `module.` (from `DDP`) prefixes during state restoration.

---

## 3. Verification Protocol

1. **Round-Trip Test**: Save a model with randomized weights, reload into a fresh instance, and compute output difference on a fixed tensor:
   $$\max |\mathbf{y}_{\text{orig}} - \mathbf{y}_{\text{loaded}}| < 10^{-6}$$
2. **Deterministic Resumption Test**: Run 5 steps, checkpoint, resume, run 5 steps. Assert loss trajectory exactly matches 10 uninterrupted steps.
