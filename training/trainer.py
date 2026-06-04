from __future__ import annotations

import math
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel

from .arguments import TrainingArguments
from .early_stopping import EarlyStoppingCallback
from .optim_utils import (
    create_optimizer,
    create_scheduler,
    resolve_amp_dtype,
    compute_total_steps,
)
from .logging_utils import (
    init_wandb,
    log_training_metrics,
    log_eval_summary,
    log_final_summary,
    print_training_header,
    finish_wandb,
)
from .checkpointing import (
    save_checkpoint,
    resume_from_checkpoint,
    prune_checkpoints,
)


class KilatTrainer:
    """
    Custom training loop with step-based or epoch-based scheduling,
    AMP (FP16/BF16/FP32), tqdm progress bars, WandB logging, and early stopping.

    Design Philosophy
    ----------------
    This trainer exists because HuggingFace's default Trainer abstracts away too many
    training loop details, making it difficult to:
    1. Implement step-based training for large-scale pretraining where epoch boundaries
       are meaningless (datasets may be infinite or too large for single-pass).
    2. Have fine-grained control over gradient accumulation timing and loss averaging.
    3. Support modern PyTorch AMP API (torch.amp >= 2.3) with proper device-type handling.
    
    The dual training mode (steps vs epochs) is intentional:
    - Steps mode: Used when training on massive datasets (e.g., web-scale pretraining)
      where you define training duration by optimizer steps rather than data passes.
      This avoids arbitrary epoch boundaries and allows precise training budgets.
    - Epochs mode: Used for fine-tuning on fixed-size datasets where you want to
      control the number of full passes through the data.
    
    Key features
    ------------
    * Real-time progress bars via tqdm with live metrics (loss, PPL, LR, step).
    * Perplexity (PPL) computed as ``exp(loss)`` and displayed alongside loss.
      PPL is more interpretable than raw loss for language modeling tasks and
      serves as an intuitive quality metric (lower is better).
    * Mixed precision via ``torch.amp`` (PyTorch >= 2.3 API):
        - ``fp16`` -- GradScaler active for numerical stability. Required because
          FP16 has limited dynamic range and gradients can underflow/overflow.
        - ``bf16`` -- GradScaler disabled. BF16 has the same exponent range as FP32
          (8 bits), so it doesn't need loss scaling. Only available on Ampere+ GPUs.
        - ``fp32`` -- no casting, suitable for debugging or CPU training.
    * Two selectable training modes via ``TrainingArguments.training_mode``:
        - ``"steps"``  -- progress measured in optimizer steps; ideal for
          large-scale pretraining where dataset size exceeds one epoch.
        - ``"epochs"`` -- progress measured in epochs; ideal for fine-tuning
          on a fixed-size dataset.
    * Cosine learning-rate schedule with linear warmup following standard practice
      from the GPT/LLM literature (warmup prevents early gradient explosions).
    * Gradient accumulation and gradient clipping to simulate larger batch sizes
      and prevent gradient explosions, respectively.
    * Periodic evaluation + early stopping (when an eval dataset is provided).
    * Checkpoint saving compatible with ``model.save_pretrained`` for
      HuggingFace ecosystem integration.
    * Optional resume from any checkpoint (restores optimizer, scheduler, scaler state)
      to support preemption recovery in long-running jobs.
    * Optional Weights & Biases logging for experiment tracking.
    * Graceful handling of KeyboardInterrupt (saves checkpoint before exit)
      to prevent losing progress in interactive/research settings.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model instance to train. Must have standard HF interface
        (forward with input_ids/labels, return_dict=True).
    args : TrainingArguments
        Hyperparameter configuration via :class:`TrainingArguments`.
    train_dataset : Dataset
        PyTorch Dataset for training. Expected to yield dicts with
        'input_ids' and 'labels' keys.
    eval_dataset : Optional[Dataset]
        Optional PyTorch Dataset for validation. If ``None``, evaluation and
        early stopping are disabled.
    data_collator : Optional[Any]
        Optional collate function passed to ``DataLoader``. If None, uses
        default PyTorch collation (expects samples to be directly stackable).

    Example
    -------
    Step-based with BF16 (for Ampere+ pretraining):

    >>> args = TrainingArguments(
    ...     output_dir="./ckpts",
    ...     training_mode="steps",
    ...     max_steps=50_000,
    ...     precision="bf16",
    ... )
    >>> trainer = KilatTrainer(model, args, train_ds, eval_ds, collator)
    >>> trainer.train()

    Epoch-based with FP16 (for fine-tuning):

    >>> args = TrainingArguments(
    ...     output_dir="./ckpts",
    ...     training_mode="epochs",
    ...     num_train_epochs=5,
    ...     precision="fp16",
    ... )
    >>> trainer = KilatTrainer(model, args, train_ds, eval_ds, collator)
    >>> trainer.train()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        # Reproducibility: Set seed before any initialization to ensure
        # consistent parameter initialization, data shuffling, and dropout patterns.
        # Manual seed is used instead of torch.manual_seed(0) to allow user control
        # over experiment reproducibility across runs.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Select available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Resolve AMP dtype and settings based on chosen precision.
        # The resolution logic encapsulates hardware capability checks:
        # - bf16: only available on CUDA 11+ and Ampere+ GPUs
        # - fp16: universally available on CUDA
        # - fp32: always available (no AMP)
        self._amp_dtype: Optional[torch.dtype] = resolve_amp_dtype(args.precision)
        self._amp_enabled: bool = self._amp_dtype is not None
        
        # GradScaler is only enabled for FP16 because BF16 has sufficient
        # dynamic range (8-bit exponent like FP32) to handle gradient values
        # without scaling. Using GradScaler with BF16 would add unnecessary
        # overhead and can actually degrade performance.
        self._scaler_enabled: bool = args.precision == "fp16"

        # Determine device_type for torch.amp.autocast context manager.
        # Even though 'cuda' is passed as string, PyTorch uses it to dispatch
        # to the correct backend. CPU AMP is not supported for training.
        self._autocast_device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Training DataLoader: shuffle=True for training to prevent the model
        # from learning dataset order patterns. pin_memory speeds up CPU->GPU
        # transfers by using pinned (page-locked) memory.
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            pin_memory=torch.cuda.is_available(),
        )

        # Evaluation DataLoader (optional) + early stopping callback
        # Early stopping uses a patience-based approach: training stops if
        # eval loss doesn't improve for 'patience' consecutive evaluations.
        # Threshold prevents stopping on negligible improvements.
        if self.eval_dataset is not None:
            self.eval_dataloader: Optional[DataLoader] = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,  # No shuffling for evaluation - deterministic results
                collate_fn=self.data_collator,
                pin_memory=torch.cuda.is_available(),
            )
            self.early_stopping: Optional[EarlyStoppingCallback] = EarlyStoppingCallback(
                patience=args.early_stopping_patience,
                threshold=args.early_stopping_threshold,
            )
        else:
            self.eval_dataloader = None
            self.early_stopping = None

        # Compute total steps: In steps mode, this is simply max_steps.
        # In epochs mode, this is calculated from num_epochs * batches_per_epoch,
        # divided by gradient_accumulation_steps since optimizer only steps
        # after accumulation is complete.
        self.total_steps = compute_total_steps(
            args.training_mode,
            args.max_steps,
            args.num_train_epochs,
            len(self.train_dataloader),
            args.gradient_accumulation_steps,
        )

        # Optimizer, scheduler, and AMP GradScaler
        # Using AdamW by default (decoupled weight decay) which is standard
        # for transformer training per "Decoupled Weight Decay Regularization"
        # (Loshchilov & Hutter, 2019).
        self.optimizer = create_optimizer(
            self.model,
            args.learning_rate,
            args.weight_decay,
        )
        
        # Cosine schedule with linear warmup: warmup prevents early training
        # instability when the model is far from optimum, while cosine decay
        # provides smooth LR reduction following the Loshchilov & Hutter (2017)
        # "SGDR" paper approach adapted for Adam.
        self.scheduler = create_scheduler(
            self.optimizer,
            self.total_steps,
            args.warmup_steps,
        )

        # GradScaler device must match the model device. 'cuda' string is used
        # even for non-CUDA devices because the scaler handles this gracefully
        # (it's a no-op on CPU but still needs the parameter for API consistency).
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=self._scaler_enabled
        )

        # Global training state (can be restored from checkpoint)
        # These track the training progress across potential restarts.
        # initial values represent a fresh training start.
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_eval_loss: float = float("inf")  # Lower is better
        self.start_time: float = time.time()  # Used for throughput calculations

        # Load state from checkpoint if requested.
        # This enables preemption recovery: if a job is killed (common in SLURM/cluster
        # environments), the trainer can resume from the last checkpoint, restoring
        # model weights, optimizer state, scheduler state, and training metrics.
        if args.resume_from_checkpoint is not None:
            self.global_step, self.current_epoch, self.best_eval_loss = resume_from_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.early_stopping,
                args.resume_from_checkpoint,
                self.device,
            )

        # Initialize WandB (if configured and library is available)
        # We pass model_config to WandB to capture architecture details
        # automatically for experiment tracking and reproducibility.
        init_wandb(
            args.report_to,
            args.run_name,
            {
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "total_steps": self.total_steps,
                "training_mode": args.training_mode,
                "precision": args.precision,
            },
            self.model.config.to_dict(),
        )

    # -------------------------------------------------------------------
    # Main training loop -- dispatcher
    # -------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the training loop according to the selected ``training_mode``.
        
        The dispatcher pattern separates step-based and epoch-based logic
        while maintaining a single public API. This is cleaner than having
        conditionals throughout the training loop.
        """
        print_training_header(
            self.args.output_dir,
            self.args.save_checkpoints,
            self.args.training_mode,
            self.total_steps,
            self.args.num_train_epochs,
            self.args.per_device_train_batch_size,
            self.args.gradient_accumulation_steps,
            self.args.learning_rate,
            self.args.warmup_steps,
            self.args.weight_decay,
            self.args.max_grad_norm,
            self.device,
            self.args.precision,
            self._scaler_enabled,
            self.args.report_to,
            self.args.seed,
        )
        self.model.train()

        try:
            if self.args.training_mode == "steps":
                self._train_by_steps()
            else:
                self._train_by_epochs()
        except KeyboardInterrupt:
            # Graceful interrupt handling: saves progress before re-raising.
            # This is critical for interactive development/research where users
            # may interrupt training to adjust hyperparameters or because of
            # resource constraints. The checkpoint preserves all state so
            # training can be resumed exactly where it left off.
            print(f"\n{'='*60}")
            print(f"Training interrupted by user at step {self.global_step}")
            print(f"Saving checkpoint before exit...")
            print(f"{'='*60}")
            self._save_checkpoint(self.global_step, tag="interrupted")
            self._finish()
            raise

    # -------------------------------------------------------------------
    # Step-based training loop
    # -------------------------------------------------------------------

    def _train_by_steps(self) -> None:
        """
        Training loop that stops exactly after ``max_steps`` optimizer steps.
        
        This mode is designed for large-scale pretraining where:
        1. The dataset may be infinite (streaming) or too large for epoch counting
        2. Training duration is measured in optimizer updates, not data passes
        3. You want precise control over the total number of optimization steps
        
        The loop iterates through the dataloader indefinitely, cycling through
        epochs as needed, until the step budget is exhausted.
        
        Key design decision: Loss averaging is done over gradient accumulation steps,
        not over the entire training. This gives a more stable and interpretable
        loss metric that reflects the effective batch size being used.
        """
        progress_bar = tqdm(
            total=self.total_steps,
            initial=self.global_step,  # Start from checkpoint position if resuming
            desc="Training (steps)",
            dynamic_ncols=True,
            unit="step",
        )

        step_within_accum: int = 0  # Counter for gradient accumulation
        running_loss: float = 0.0  # Accumulated loss for the current accumulation window

        # Start from the epoch we were at (1 if fresh start, or restored value)
        epoch = self.current_epoch or 1
        while self.global_step < self.total_steps:
            self.current_epoch = epoch

            for batch in self.train_dataloader:
                loss_val = self._forward_backward(batch)
                running_loss += loss_val
                step_within_accum += 1

                # Only step the optimizer after accumulating enough gradients.
                # This simulates larger batch sizes without increasing memory usage.
                # For example, batch_size=8 with accumulation_steps=4 gives
                # effective batch size of 32.
                if step_within_accum == self.args.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1
                    step_within_accum = 0  # Reset accumulation counter
                    current_lr = self.scheduler.get_last_lr()[0]

                    # Compute average loss over the accumulation window.
                    # We divide by accumulation steps to get the mean loss per
                    # micro-batch, which is more interpretable than the sum.
                    avg_loss = running_loss / self.args.gradient_accumulation_steps
                    
                    # PPL calculation: exp(loss) for language modeling tasks.
                    # We cap at loss=100 to prevent overflow (exp(100) ≈ 2.7e43).
                    # In practice, any loss > 10 is already catastrophic for LM.
                    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ppl": f"{ppl:.1f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )
                    running_loss = 0.0

                    # Periodic logging: Log metrics at configurable intervals.
                    # This prevents overwhelming the logging backend with too many
                    # data points while still capturing the training trajectory.
                    if self.global_step % self.args.logging_steps == 0:
                        log_training_metrics(
                            self.global_step,
                            self.total_steps,
                            avg_loss,
                            ppl,
                            current_lr,
                            grad_norm,
                            self.current_epoch,
                            self.start_time,
                            self.args.report_to,
                        )

                    # Periodic evaluation: Only if eval dataset exists and at
                    # specified intervals. We don't evaluate too frequently because
                    # evaluation is expensive (full pass through eval set).
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        should_stop = self._run_eval_and_check_stopping()
                        if should_stop:
                            progress_bar.close()
                            return

                    # Periodic checkpoint: Save model state at regular intervals.
                    # This provides fault tolerance and allows model selection
                    # across different training stages.
                    if self.args.save_checkpoints and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    # Check if total steps reached: This check is placed here
                    # (rather than at the start of the loop) to ensure we
                    # complete the current optimization step before stopping.
                    if self.global_step >= self.total_steps:
                        print(f"\n{'='*60}")
                        print(f"Training complete ({self.total_steps:,} steps)")
                        print(f"{'='*60}")
                        if self.args.save_checkpoints:
                            self._save_checkpoint(self.global_step, tag="final")
                        progress_bar.close()
                        self._finish()
                        return

            epoch += 1

        progress_bar.close()

    # -------------------------------------------------------------------
    # Epoch-based training loop
    # -------------------------------------------------------------------

    def _train_by_epochs(self) -> None:
        """
        Training loop that stops after ``num_train_epochs`` full epochs.
        
        This mode is designed for fine-tuning and smaller datasets where:
        1. The dataset size is known and fixed
        2. You want to control the number of full passes through the data
        3. Each epoch represents a complete pass through the training set
        
        Key difference from steps mode: The total_steps is calculated from
        epochs * batches_per_epoch, providing an equivalent step budget for
        the scheduler, but the loop structure is epoch-based for clarity.
        
        Design decision: We still support eval/checkpoint at step intervals
        within epochs, not just at epoch boundaries, because for large datasets,
        waiting for a full epoch to evaluate could mean waiting hours.
        """
        start_epoch = self.current_epoch or 1

        for epoch in range(start_epoch, self.args.num_train_epochs + 1):
            self.current_epoch = epoch

            # Each epoch gets its own progress bar for cleaner visualization.
            # 'leave=True' keeps completed epoch bars visible for reference.
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch}/{self.args.num_train_epochs}",
                dynamic_ncols=True,
                unit="batch",
                leave=True,
            )

            epoch_loss: float = 0.0  # Running loss for this epoch
            step_within_accum: int = 0

            for batch_idx, batch in progress_bar:
                loss_val = self._forward_backward(batch)
                epoch_loss += loss_val
                step_within_accum += 1

                if step_within_accum == self.args.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1
                    step_within_accum = 0

                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    # Average loss over all batches processed so far in this epoch.
                    # This gives a running estimate of epoch-level performance.
                    avg_loss = epoch_loss / (batch_idx + 1)
                    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "ppl": f"{ppl:.1f}",
                            "lr": f"{current_lr:.2e}",
                            "step": self.global_step,
                        }
                    )

                    # Periodic logging within epoch: Same logging logic as steps mode
                    if self.global_step % self.args.logging_steps == 0:
                        log_training_metrics(
                            self.global_step,
                            self.total_steps,
                            avg_loss,
                            ppl,
                            current_lr,
                            grad_norm,
                            self.current_epoch,
                            self.start_time,
                            self.args.report_to,
                        )
                        epoch_loss = 0.0  # Reset for next logging interval

                    # Periodic evaluation within epoch
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        should_stop = self._run_eval_and_check_stopping()
                        if should_stop:
                            return

                    # Periodic checkpoint within epoch
                    if self.args.save_checkpoints and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(self.global_step)

            # End of epoch: Always evaluate at epoch boundaries to get
            # a complete picture of model performance on the full dataset.
            # This is important even if we evaluated mid-epoch.
            print(f"\n[Epoch {epoch}] Complete.")

            if self.eval_dataloader is not None:
                should_stop = self._run_eval_and_check_stopping()
                if should_stop:
                    return

            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag=f"epoch-{epoch}")

        # All epochs complete
        print(f"\n{'='*60}")
        print(f"Training complete ({self.args.num_train_epochs} epochs, {self.global_step:,} steps)")
        print(f"{'='*60}")
        if self.args.save_checkpoints:
            self._save_checkpoint(self.global_step, tag="final")
        self._finish()


    def _forward_backward(self, batch: dict[str, torch.Tensor]) -> float:
        """
        Run a single forward + backward pass with AMP autocast.
        
        Returns the loss value BEFORE scaling and accumulation division,
        so callers can accumulate it properly. The returned loss is
        multiplied by gradient_accumulation_steps to reflect the original
        (pre-scaling) loss value.
        
        Key design decisions:
        - Loss scaling (dividing by gradient_accumulation_steps) happens
          BEFORE backward. This ensures the accumulated gradient is the
          mean of micro-batch gradients, not the sum. This is equivalent
          to training with a larger batch size.
        - Non-blocking transfers (non_blocking=True) overlap data movement
          with computation, hiding CPU->GPU transfer latency.
        - autocast handles the mixed precision conversion automatically
          based on the configured dtype and device type.
        """
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # AMP autocast context: Automatically casts operations to the
        # specified precision where beneficial (e.g., matmul in FP16/BF16)
        # while keeping sensitive operations (e.g., softmax, layernorm) in FP32.
        # This follows the "Mixed Precision Training" paper (Micikevicius et al., 2018).
        with torch.amp.autocast(
            device_type=self._autocast_device,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
            # Normalize loss by gradient accumulation steps to get mean gradient.
            # This ensures consistent loss interpretation regardless of accumulation.
            loss = outputs.loss / self.args.gradient_accumulation_steps

        # Scale the loss before backward for FP16 training stability.
        # In BF16/FP32 modes, scaler.scale is a no-op.
        self.scaler.scale(loss).backward()

        # Return the original (unscaled) loss for logging/metrics.
        # Multiplying by accumulation steps recovers the true per-batch loss.
        return loss.item() * self.args.gradient_accumulation_steps

    def _optimizer_step(self) -> torch.Tensor:
        """
        Unscale, clip gradients, then execute optimizer + scheduler step.
        
        Returns the gradient norm before clipping for monitoring purposes.
        
        The step sequence is critical for correct FP16 training:
        1. unscale_: Reverses the loss scaling to get true gradients
        2. clip_grad_norm_: Prevents gradient explosion (common in transformers)
        3. scaler.step: Updates weights (may skip if gradients contain infs/nans)
        4. scaler.update: Adjusts the loss scale for next iteration
        5. scheduler.step: Updates learning rate
        6. zero_grad: Clears gradients for next accumulation cycle
        
        set_to_none=True is used instead of zero_() for memory efficiency:
        it releases gradient tensors entirely rather than filling with zeros.
        """
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.args.max_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm

    def _run_eval_and_check_stopping(self) -> bool:
        """
        Run evaluation, update best checkpoint, and check early stopping.
        
        Returns True if early stopping triggered and training should stop.
        
        Side effects:
        - Sets self.best_eval_loss if current eval loss is better
        - Saves best checkpoint (if save_checkpoints enabled)
        - Logs early stopping message and saves final checkpoint if triggered
        """
        eval_loss, eval_ppl = self.evaluate()
        
        # Critical: Set model back to training mode after evaluation.
        # evaluate() sets model.eval(), which disables dropout and batch norm.
        # Forgetting this would silently degrade training quality.
        self.model.train()

        # Track best model: Only save checkpoint when we find a new best.
        # This prevents checkpoint bloat and ensures we always have access
        # to the best-performing model state.
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag="best")

        # Early stopping check: Uses patience-based approach.
        # The callback tracks consecutive evaluations without improvement
        # and signals stop when patience is exhausted. This prevents
        # stopping on temporary loss spikes while still preventing overfitting.
        if self.early_stopping and self.early_stopping.check(eval_loss):
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at step {self.global_step}")
            print(f"{'='*60}")
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag="early-stopped")
            self._finish()
            return True
        return False


    @torch.inference_mode()
    def evaluate(self) -> tuple[float, float]:
        """
        Run full evaluation pass over the eval dataset.
        
        Uses inference_mode() instead of no_grad() because:
        - inference_mode() provides additional optimizations by disabling
          autograd version tracking entirely, not just gradient computation.
        - This gives a slight speedup (~5-10%) for evaluation.
        
        Returns (average_loss, perplexity) tuple.
        """
        self.model.eval()
        eval_loss: float = 0.0

        eval_progress = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            dynamic_ncols=True,
            unit="batch",
            leave=False,  # Don't leave progress bar after completion
        )

        for batch in eval_progress:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self._autocast_device,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
                eval_loss += outputs.loss.item()

            # Update progress bar with current loss for real-time monitoring
            eval_progress.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        # Compute average loss: divide by number of batches.
        # max(1, ...) prevents division by zero in edge case of empty dataloader.
        avg_eval_loss = eval_loss / max(1, len(self.eval_dataloader))
        eval_ppl = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float("inf")

        log_eval_summary(
            avg_eval_loss,
            eval_ppl,
            self.global_step,
            self.best_eval_loss,
            self.args.report_to,
        )

        return avg_eval_loss, eval_ppl

    def _save_checkpoint(self, step: int, tag: Optional[str] = None) -> None:
        """
        Save training checkpoint with pruning.
        
        The pruning mechanism (save_total_limit) prevents unbounded disk usage
        by keeping only the N most recent checkpoints. This is critical for
        long-running training where checkpoints can be gigabytes each.
        
        Checkpoint contents include:
        - Model weights (via save_pretrained for HF compatibility)
        - Optimizer state (for exact training resumption)
        - Scheduler state (to continue LR schedule from same point)
        - AMP scaler state (for FP16 training stability continuity)
        - Training metrics (global_step, epoch, best_loss)
        - Early stopping state (for correct patience counting across restarts)
        """
        if not self.args.save_checkpoints:
            return

        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.global_step,
            self.current_epoch,
            self.best_eval_loss,
            self.early_stopping,
            self.args.output_dir,
            step,
            tag,
        )
        prune_checkpoints(self.args.output_dir, self.args.save_total_limit)

    def _finish(self) -> None:
        """
        Final cleanup: log summary metrics and close WandB.
        
        This is called in all exit paths (normal completion, early stopping,
        interruption) to ensure consistent logging and resource cleanup.
        """
        log_final_summary(
            self.global_step,
            self.start_time,
            self.best_eval_loss,
            self.args.output_dir,
        )
        finish_wandb(self.args.report_to)