---
name: pytorch-training
description: Deep PyTorch ML Training Engineering for Kilat. Covers autograd mechanics, forward/backward correctness, gradient accumulation, optimizer/scheduler lifecycle, mixed-precision (AMP/BF16/FP16), and numerical stability.
---

# PyTorch Training Engineering Skill

## 1. When to Activate This Skill
Activate this skill whenever the task involves:
- Writing, modifying, or refactoring training loops (`KilatTrainer`, custom loops).
- Modifying optimizer instantiation, learning rate schedulers, or warmup schedules.
- Gradient accumulation, gradient clipping, or gradient norm tracking.
- Mixed precision training (`torch.autocast`, `GradScaler`, BF16 vs FP16).
- Diagnosing numerical instability (NaN/Inf losses, exploding gradients, silent underflow).
- Switching between `model.train()` and `model.eval()`, or managing dropout and norm statistics.

---

## 2. Core Engineering Invariants

### 2.1 Optimization & Scheduler Step Ordering
- **Rule**: `optimizer.step()` must **always** occur before `lr_scheduler.step()`.
- **AMP Scaler Rule**: When using `torch.cuda.amp.GradScaler`, the sequence must strictly be:
  ```python
  scaler.scale(loss).backward()
  if (step + 1) % gradient_accumulation_steps == 0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      optimizer.zero_grad(set_to_none=True)
  ```
- **Anti-Pattern**: Calling `scheduler.step()` before `scaler.step(optimizer)`. If the scaler skips the optimizer step due to Inf/NaN gradients, stepping the scheduler causes premature learning rate decay on invalid updates.

### 2.2 Gradient Accumulation Correctness
- **Loss Scaling**: Loss must be scaled by accumulation factor: `loss = raw_loss / gradient_accumulation_steps` before `.backward()`.
- **Zero Grad Timing**: `optimizer.zero_grad(set_to_none=True)` must only be called at the boundary of `gradient_accumulation_steps`, never inside every micro-batch.
- **Memory Efficiency**: Always prefer `optimizer.zero_grad(set_to_none=True)` over `set_to_none=False` to free gradient buffers and eliminate zero-fill memory bandwidth.

### 2.3 Numerical Stability & Precision Dynamics
- **BF16 vs FP16**:
  - `torch.bfloat16`: 8-bit dynamic range (matches FP32 exponent), does **not** require a `GradScaler`. Highly preferred on Ampere+ (RTX 30xx/40xx, A100, H100).
  - `torch.float16`: 5-bit dynamic range, **strictly requires** `torch.cuda.amp.GradScaler` to prevent gradient underflow.
- **NaN/Inf Guard**:
  ```python
  if not torch.isfinite(loss):
      raise FloatingPointError(f"Training diverged with non-finite loss: {loss.item()}")
  ```
- **Logit Clamping**: When computing cross-entropy or softmax on large vocabularies with hybrid attention, ensure logits are bounded or stable before loss calculation.

### 2.4 Mode Invariance
- Training: Ensure `model.train()` is enabled so dropout and training paths (e.g. MoE auxiliary load loss) operate correctly.
- Evaluation: Wrap validation in `model.eval()` and `torch.no_grad()` or `torch.inference_mode()`. Never evaluate with gradients enabled.

---

## 3. Verification Protocol

Before completing any change in this domain:
1. **Gradient Check**: Assert that 100% of trainable parameters receive valid non-zero gradients:
   ```python
   for name, param in model.named_parameters():
       if param.requires_grad:
           assert param.grad is not None, f"Missing gradient for {name}"
           assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"
   ```
2. **Overfit-One-Batch Test**: Feed a single micro-batch for 20-50 steps. The loss must decrease monotonically to near-zero ($< 0.1$). If loss plateaus or diverges, autograd or optimizer mechanics are broken.
