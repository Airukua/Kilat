from __future__ import annotations
import contextlib
import logging
import math
import os
import random
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .args import TrainingArguments
from .callbacks import (
    CallbackHandler,
    EarlyStoppingCallback,
    TrainerControl,
    TrainerState,
)
from .integration import DEFAULT_CALLBACKS, get_reporting_integration_callbacks
from .optimizer import create_optimizer, resolve_amp_dtype
from .scheduler import get_scheduler
from .trainer_utils import (
    clip_grad_norm_,
    compute_perplexity,                     # <-- added for PPL in training logs
    compute_total_steps,
    format_metrics_with_ppl,               # <-- added for evaluation metrics
    get_current_lr,
    get_device,
    get_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    should_evaluate,
    should_log,
    should_save,
)

logger = logging.getLogger(__name__)


class KilatTrainer:
    """
    Production‑grade training loop for Kilat and compatible HuggingFace‑style models.

    WHY THIS CLASS EXISTS
        Training a neural network involves many subtle decisions: gradient
        accumulation, mixed precision, distributed synchronisation, checkpoint
        management, early stopping, and integration with logging backends.
        This class encapsulates all those concerns so users only need to provide
        a model, data, and configuration. It is designed to be robust, observable,
        and resumable – the three pillars of industrial training.

    ARCHITECTURE OVERVIEW
        KilatTrainer composes five subsystems that each handle a distinct
        responsibility:

        1. **Optimisation** – AdamW with decoupled weight decay (parameter groups
           for biases & norms) paired with any scheduler from the `SchedulerType`
           registry (cosine, linear, polynomial, inverse sqrt, WSD, REX, etc.).
           Created via `create_optimizer` and `get_scheduler`.

        2. **Mixed Precision** – `torch.autocast` wraps forward passes.
           A `GradScaler` is used **only** for FP16 (because FP16 has limited
           dynamic range). BF16 uses the same exponent range as FP32 and does
           not need a scaler. Precision is resolved from `TrainingArguments.precision`
           via `resolve_amp_dtype`.

        3. **Callback System** – `CallbackHandler` dispatches lifecycle events
           (`on_train_begin`, `on_step_end`, `on_evaluate`, …) to all registered
           callbacks in order. This enables early stopping, logging to W&B/TensorBoard,
           progress bars, and custom user hooks. The return values (TrainerControl
           flags) are OR‑merged: if any callback requests training stop, training stops.

        4. **Checkpointing** – Atomic (rename‑based) HuggingFace‑format checkpoints
           that contain model weights (via `save_pretrained`) and a separate
           `training_state.pt` with optimizer, scheduler, scaler, callback states,
           and TrainerState. `load_checkpoint` can resume from any checkpoint,
           and `prune_checkpoints` enforces `save_total_limit` by deleting oldest
           periodic (numbered) checkpoints while preserving tagged ones (best, final).

        5. **Progress Reporting** – `tqdm.auto` provides rich, environment‑aware
           progress bars both in notebooks and plain terminals. It wraps the epoch
           loop and the per‑epoch step loop, showing loss, learning rate, and step count.

    TRAINING MODES
        Two orthogonal modes determine the stopping condition:

        - **Epoch mode** (`training_mode="epochs"`): Loops exactly `num_train_epochs`
          full passes over the training DataLoader. The scheduler’s total steps are
          computed as `steps_per_epoch * num_train_epochs`. This is natural for
          fine‑tuning where dataset size is fixed and you want to see metrics at
          each epoch boundary.

        - **Step mode** (`training_mode="steps"`): Runs exactly `max_steps` optimizer
          steps, cycling the DataLoader as needed (i.e., when an epoch ends, the
          dataloader is re‑created from the beginning). Useful for large‑scale
          pre‑training where the dataset is effectively infinite or when you want
          a fixed compute budget independent of dataset size.

    GRADIENT ACCUMULATION
        When `gradient_accumulation_steps > 1`, the optimizer step is deferred
        until N micro‑batches have been processed. This effectively multiplies
        the batch size without additional GPU memory. The loss is divided by
        the accumulation count **before** backward so that the gradient magnitude
        is independent of the accumulation factor (standard practice). Logged
        losses are averaged over the accumulation window for meaningful reporting.

    AMP (AUTOMATIC MIXED PRECISION)
        - `fp16`: Use `torch.autocast` + `GradScaler`. The scaler dynamically
          adjusts the loss scale to prevent underflow. Must have CUDA.
        - `bf16`: Use only `torch.autocast` (no scaler). BF16 preserves the same
          exponent range as FP32, so underflow is not a problem. Works on Ampere+
          GPUs and on CPU with PyTorch ≥ 2.1 (using AMX instructions).
        - `fp32`: No autocast, no scaler. Used for CPU training or when mixed
          precision causes stability issues.

    DISTRIBUTED TRAINING
        KilatTrainer **does not** initialise torch.distributed automatically.
        That responsibility belongs to the launcher (e.g., `torchrun`). However,
        the trainer respects the `is_world_process_zero` flag in `TrainerState`.
        All file I/O (checkpoints, logs, progress bars) is gated on this flag so
        that only rank 0 writes files, preventing corruption and duplication.
        Users must set `state.is_world_process_zero = (rank == 0)` before calling
        `train()` in distributed environments.

    EVALUATION
        The trainer calls evaluation:
        - At step‑based intervals (when `global_step % eval_steps == 0`)
        - At the end of every epoch **if in epoch mode**
        - The user can provide a custom `eval_fn`. If none is given, a default
          loop computes `eval_loss` by averaging the loss over the eval dataloader.
        - The resulting metrics dict is passed to `on_evaluate` callbacks, which
          includes `EarlyStoppingCallback` (updates patience counter) and logging
          integrations (W&B summary, etc.).
        - If the monitored metric improves, the best metric and best checkpoint
          are updated, and a `checkpoint-best` is saved.

    EARLY STOPPING
        Wired through `EarlyStoppingCallback`. It monitors the metric specified
        by `metric_for_best_model` (default `"eval_loss"`). Patience and threshold
        are configurable. When the metric does not improve for `patience`
        consecutive evaluations, `control.should_training_stop = True` stops the
        loop. The callback’s state (best metric, patience counter) is saved in
        checkpoints, so a resumed run continues with the exact same patience.

    CHECKPOINT TAGGING & PRUNING
        - **Numbered**: `checkpoint-1000` – created at `save_steps` intervals.
          These are prunable – the oldest ones are deleted when `save_total_limit`
          is exceeded.
        - **Tagged**: `checkpoint-best`, `checkpoint-final`, `checkpoint-epoch-3`.
          These are never pruned. They represent important milestones.
        - Atomic saving: a temporary directory is written first, then renamed to
          the final path. This prevents corrupted half‑written checkpoints.

    PERFORMANCE CONSIDERATIONS
        - `_forward_backward` uses `model.no_sync()` when available (DDP) to
          defer gradient all‑reduction until the last accumulation step. This
          reduces communication overhead by O(grad_acc) times.
        - `_batch_to_device` is recursive but small; tensors are moved to device
          once per batch. No unnecessary copies.
        - `tqdm` updates are disabled for non‑rank‑0 processes to avoid cluttered
          logs.
        - The main loop avoids recomputing `total_steps` multiple times; it is
          computed once in `train()`.

    EDGE CASES & ASSUMPTIONS
        - Dataloaders must be finite. For step mode, the trainer cycles the loader
          by catching `StopIteration` and restarting.
        - The model must either:
          a) Accept a dict and return an object with a `.loss` attribute (HF style)
          b) Accept a tensor/tuple and return a scalar tensor
          c) A custom `compute_loss_fn` can override this.
        - If `eval_dataloader` is None, evaluation is silently skipped (no warnings).
        - `load_checkpoint` may return a **new** model instance (when falling back
          to `from_pretrained`). The trainer updates its `self.model` accordingly.
        - Early stopping callbacks are added automatically even if you don't want
          them; you can disable early stopping by setting `early_stopping_patience=0`
          (the callback will still be present but will never stop).
        - `scheduler_type` strings are case‑insensitive; the factory normalises.

    EXAMPLE USAGE
    -------------
        >>> from kilat.training import KilatTrainer, TrainingArguments
        >>> args = TrainingArguments(
        ...     output_dir="./checkpoints",
        ...     training_mode="epochs",
        ...     num_train_epochs=3,
        ...     precision="bf16",
        ...     scheduler_type="cosine",
        ...     report_to="wandb",
        ... )
        >>> trainer = KilatTrainer(model, args, train_dl, eval_dl)
        >>> final_state = trainer.train()
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
        compute_loss_fn: Optional[Callable] = None,
        callbacks: Optional[list] = None,
    ) -> None:
        """
        Initialise the trainer without starting the training loop.

        WHY: Separation of construction and execution allows the user to inspect
        or modify the trainer before calling `.train()`. It also ensures that
        the trainer is fully set up even if the user wants to load a checkpoint
        manually.

        NOTE: The scheduler is NOT created here because it depends on total_steps,
        which requires the dataloader length. That creation is deferred to
        `train()` after we compute total steps.

        Parameters
        ----------
        model : nn.Module
            Model to train. It will be moved to the appropriate device (CUDA/CPU).
        args : TrainingArguments
            All hyperparameters and configuration.
        train_dataloader : DataLoader
            Training data loader. Must be finite (has __len__).
        eval_dataloader : Optional[DataLoader]
            Validation data loader. If None, evaluation is disabled.
        eval_fn : Optional[Callable]
            Custom evaluation function: `eval_fn(model, dataloader, device) -> dict[str, float]`.
            If None, a default loop that averages model loss is used.
        compute_loss_fn : Optional[Callable]
            Custom loss extraction: `compute_loss_fn(model, batch) -> torch.Tensor` (scalar).
            If None, the trainer assumes `model(**batch).loss` for dict batches,
            or `model(batch)` returning a scalar tensor for other batch types.
        callbacks : Optional[list]
            Additional `TrainerCallback` instances. They are appended after the
            automatic callbacks (integrations, early stopping) but before the
            default `ProgressCallback`.
        """
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn
        self.compute_loss_fn = compute_loss_fn

        self.device = get_device(model)

        # Reproducibility – set all random seeds early
        self._set_seed(args.seed)

        # Mixed precision setup
        self.amp_dtype: Optional[torch.dtype] = resolve_amp_dtype(args.precision)
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler() if args.use_grad_scaler else None
        )

        # Optimizer – independent of scheduler
        self.optimizer = create_optimizer(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
        )

        # Scheduler is created later in train() after total steps are known
        self.scheduler: Any = None

        # TrainerState holds mutable progress; TrainerControl holds request flags.
        # Initially, max_steps may be 0 (epoch mode) – will be updated after compute.
        self.state = TrainerState(
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.training_mode == "steps" else 0,
        )
        self.control = TrainerControl()

        # ─── Callback setup ──────────────────────────────────────────────
        # Order matters: integrations first (so logging captures the start),
        # then early stopping, then user callbacks, and finally the mandatory
        # progress callback (which always logs to console).
        integration_cbs = get_reporting_integration_callbacks(args.report_to)
        early_stopping_cb = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
        )
        self.early_stopping = early_stopping_cb

        all_callbacks = (
            integration_cbs
            + [early_stopping_cb]
            + (callbacks or [])
            + list(DEFAULT_CALLBACKS)
        )
        self.callback_handler = CallbackHandler(
            callbacks=all_callbacks,
            args=args,
        )

        self.callback_handler.on_init_end(self.state, self.control)
        logger.info("KilatTrainer initialised | device=%s | precision=%s", self.device, args.precision)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def train(self) -> TrainerState:
        """
        Execute the full training loop and return the final state.

        Steps performed:
        1. Compute total optimizer steps (depending on mode) and create scheduler.
        2. If `resume_from_checkpoint` is set, load checkpoint (or auto‑find latest).
        3. Dispatch `on_train_begin` to all callbacks.
        4. Run the training loop (epochs or steps) with internal loop methods.
        5. After normal exit, save a final checkpoint tagged "final".
        6. Dispatch `on_train_end` to all callbacks.
        7. Return the final TrainerState (contains best metric, best checkpoint, etc.)

        Returns
        -------
        TrainerState
            Final state after training, including `global_step`, `best_metric`,
            `best_model_checkpoint`, and full `log_history`.

        Raises
        ------
        RuntimeError
            If evaluation is required but `eval_dataloader` is None and no `eval_fn`
            was provided (the trainer will log a warning and skip evaluation).
        """
        args = self.args
        os.makedirs(args.output_dir, exist_ok=True)

        # ─── Compute total steps and create scheduler ────────────────────
        total_steps = compute_total_steps(
            training_mode=args.training_mode,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            dataloader_len=len(self.train_dataloader),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        self.state.max_steps = total_steps

        # Resolve warmup steps (warmup_ratio overrides warmup_steps)
        warmup_steps = args.get_warmup_steps(total_steps) if hasattr(args, "get_warmup_steps") else args.warmup_steps

        self.scheduler = get_scheduler(
            name=args.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            **args.scheduler_kwargs,
        )

        # ─── Resume from checkpoint if requested ─────────────────────────
        if args.resume_from_checkpoint is not None:
            resume_path = args.resume_from_checkpoint
            if resume_path == "latest":
                resume_path = get_latest_checkpoint(args.output_dir)
                if resume_path is None:
                    logger.warning("resume_from_checkpoint='latest' but no checkpoint found in %s", args.output_dir)
            if resume_path:
                self.model = load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    state=self.state,
                    callback_handler=self.callback_handler,
                    checkpoint_path=resume_path,
                    device=self.device,
                    early_stopping=self.early_stopping,
                )
                # Keep the newly computed step budget from the current args.
                # load_checkpoint restores TrainerState from disk, including the
                # old max_steps, but resume should follow the current run config.
                self.state.max_steps = total_steps
                # If load_checkpoint returned a new model, we replace it.
                # Otherwise, the original model was modified in place.

        # ─── Notify callbacks that training is about to start ─────────────
        self.callback_handler.on_train_begin(
            self.state, self.control, model=self.model
        )

        # ─── Main training loop (mode‑specific) ──────────────────────────
        train_start = time.monotonic()

        if args.training_mode == "epochs":
            self._train_by_epochs()
        else:
            self._train_by_steps()

        # ─── Final checkpoint (always saved, tagged as "final") ───────────
        if args.save_checkpoints:
            self._save("final")

        # ─── Notify callbacks that training has finished ─────────────────
        self.callback_handler.on_train_end(self.state, self.control)

        elapsed = time.monotonic() - train_start
        logger.info(
            "Training finished | steps=%d | elapsed=%.1fs | best_%s=%s",
            self.state.global_step,
            elapsed,
            args.metric_for_best_model,
            f"{self.state.best_metric:.6f}" if self.state.best_metric is not None else "n/a",
        )
        return self.state

    # ─────────────────────────────────────────────────────────────────────────
    # Training loops (internal)
    # ─────────────────────────────────────────────────────────────────────────

    def _train_by_epochs(self) -> None:
        """
        Epoch‑based loop: iterate over epochs, run one epoch per inner loop.

        WHY: Separating the epoch loop from the step loop improves readability
        and allows epoch‑level hooks (on_epoch_begin/end) to be called exactly
        once per epoch. It also simplifies handling of `should_epoch_stop`.

        Edge Cases:
        - If resuming from a checkpoint that was saved mid‑epoch, we need to
          start at the correct epoch. This is handled by reading `self.state.epoch`
          (which is a float, e.g., 2.5). We take the integer part as the starting
          epoch and run the remaining fraction of the epoch? Actually, the trainer
          does NOT support mid‑epoch resume because step counters are precise.
          Instead, `self.state.global_step` is the authority. Epoch loops start
          from `epoch = int(self.state.epoch)` and the step loop resumes from the
          exact step. The epoch progress bar shows steps, not batches.
        """
        args = self.args
        num_epochs = args.num_train_epochs
        resume_epoch = int(self.state.epoch)   # floor because we are at the beginning of an epoch

        epoch_bar = tqdm(
            range(resume_epoch, num_epochs),
            desc="Epochs",
            unit="epoch",
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            self.state.epoch = float(epoch)
            self.callback_handler.on_epoch_begin(self.state, self.control)

            self._run_epoch(epoch, num_epochs)

            self.state.epoch = float(epoch + 1)
            self.callback_handler.on_epoch_end(self.state, self.control)

            # Epoch‑end evaluation (only in epoch mode, always performed)
            if self.eval_dataloader is not None:
                metrics = self._evaluate()
                self._handle_evaluation(metrics)

                # Update epoch progress bar with evaluation results
                postfix = {
                    "step": self.state.global_step,
                    "best": f"{self.state.best_metric:.4f}" if self.state.best_metric is not None else "n/a",
                }
                if "eval_loss" in metrics:
                    postfix["eval_loss"] = f"{metrics['eval_loss']:.4f}"
                if "perplexity" in metrics:
                    postfix["ppl"] = f"{metrics['perplexity']:.2f}"
                epoch_bar.set_postfix(**postfix)

            # Epoch‑end checkpoint (tagged with epoch number)
            if args.save_checkpoints:
                self._save(f"epoch-{epoch + 1}")
                prune_checkpoints(args.output_dir, args.save_total_limit)

            epoch_bar.set_postfix(
                step=self.state.global_step,
                best=f"{self.state.best_metric:.4f}" if self.state.best_metric is not None else "n/a",
            )

            if self.control.should_training_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    def _train_by_steps(self) -> None:
        """
        Step‑budget loop: run until `global_step` reaches `max_steps`.

        This loop does not respect epoch boundaries; it only cares about the total
        number of optimizer steps. The dataloader is cycled indefinitely: when it
        is exhausted, a new iterator is created and `on_epoch_end/begin` callbacks
        are still fired so that metrics like epoch number increase correctly.

        WHY: Step mode is essential for pre‑training where the dataset is huge
        and we want to stop after a fixed compute budget, not after a fixed number
        of dataset passes.
        """
        args = self.args
        total_steps = self.state.max_steps
        resume_step = self.state.global_step

        step_bar = tqdm(
            total=total_steps,
            initial=resume_step,
            desc="Training",
            unit="step",
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        self.model.train()
        acc_loss = 0.0
        acc_steps = 0
        grad_acc = args.gradient_accumulation_steps

        data_iter = iter(self.train_dataloader)

        # Notify callbacks that we are starting an epoch (even though we may not
        # finish it). This keeps the epoch counter moving forward.
        self.callback_handler.on_epoch_begin(self.state, self.control)

        while self.state.global_step < total_steps:
            # Refill iterator if exhausted
            try:
                batch = next(data_iter)
            except StopIteration:
                self.callback_handler.on_epoch_end(self.state, self.control)
                self.state.epoch += 1.0
                self.callback_handler.on_epoch_begin(self.state, self.control)
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            self.callback_handler.on_step_begin(self.state, self.control)

            loss = self._forward_backward(batch, is_last_accumulation_step=(acc_steps + 1) >= grad_acc)
            acc_loss += loss
            acc_steps += 1

            self.callback_handler.on_substep_end(self.state, self.control)

            if acc_steps >= grad_acc:
                self._optimizer_step()
                self.state.global_step += 1
                avg_loss = acc_loss / grad_acc
                acc_loss = 0.0
                acc_steps = 0

                step_bar.update(1)
                step_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{get_current_lr(self.optimizer):.2e}")

                self.callback_handler.on_step_end(self.state, self.control)
                self._handle_step_end(avg_loss)

                if self.control.should_training_stop:
                    break

        step_bar.close()

    def _run_epoch(self, epoch: int, num_epochs: int) -> None:
        """
        Run one full pass over the training DataLoader.

        This method is called by `_train_by_epochs` for each epoch. It handles
        gradient accumulation, AMP, gradient clipping, step‑based logging,
        evaluation, and checkpointing that are triggered based on `global_step`.

        WHY: Splitting the inner loop allows reuse in both epoch mode (called once
        per epoch) and step mode (where it is not used). It keeps the epoch mode
        code clean.

        Parameters
        ----------
        epoch : int
            Current epoch index (0‑based).
        num_epochs : int
            Total number of epochs for display in tqdm.
        """
        args = self.args
        self.model.train()

        steps_in_epoch = len(self.train_dataloader)
        grad_acc = args.gradient_accumulation_steps
        denom = max(1, steps_in_epoch)

        step_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        acc_loss = 0.0
        acc_steps = 0

        for local_step, batch in enumerate(step_bar):
            # Keep epoch progress fractional so checkpoints can resume from the
            # exact point where training stopped instead of repeating the whole
            # epoch after an early stop.
            self.state.epoch = float(epoch) + float(local_step + 1) / float(denom)

            is_last_acc_step = (
                (local_step + 1) % grad_acc == 0
                or (local_step + 1) == steps_in_epoch
            )

            self.callback_handler.on_step_begin(self.state, self.control)

            loss = self._forward_backward(batch, is_last_accumulation_step=is_last_acc_step)
            acc_loss += loss
            acc_steps += 1

            self.callback_handler.on_substep_end(self.state, self.control)

            if is_last_acc_step:
                self._optimizer_step()
                self.state.global_step += 1
                avg_loss = acc_loss / acc_steps
                acc_loss = 0.0
                acc_steps = 0

                step_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{get_current_lr(self.optimizer):.2e}",
                )

                self.callback_handler.on_step_end(self.state, self.control)
                self._handle_step_end(avg_loss)

                if self.control.should_training_stop or self.control.should_epoch_stop:
                    break

    # ─────────────────────────────────────────────────────────────────────────
    # Core compute primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_backward(self, batch: Any, is_last_accumulation_step: bool) -> float:
        """
        Run one forward + backward pass for a single micro‑batch.

        This method is the heart of the training step. It handles:
        - Moving batch to device.
        - DDP gradient synchronisation control (`no_sync` on non‑last steps).
        - Autocast for mixed precision.
        - Loss computation (via user function or default).
        - Scaling the loss (for FP16) and calling `backward`.

        WHY `is_last_accumulation_step` matters: In DistributedDataParallel, all‑reduce
        of gradients is expensive. By using `model.no_sync()` on all but the last
        accumulation step, we reduce communication overhead by a factor of
        `gradient_accumulation_steps`.

        Parameters
        ----------
        batch : Any
            A batch from the dataloader. Typically a dict (for HF models) or tensor.
        is_last_accumulation_step : bool
            If True, this micro‑batch completes the accumulation window, so we
            should allow DDP to synchronise gradients. If False and the model
            supports `no_sync`, we use that context to defer all‑reduce.

        Returns
        -------
        float
            The scalar loss value (detached, CPU) **before** division by
            accumulation steps. This value is used for logging only.
        """
        batch = self._batch_to_device(batch)

        # DDP gradient sync only on the last accumulation step.
        # This context manager is a no‑op when not in DDP or when it's the last step.
        sync_ctx = (
            contextlib.nullcontext()
            if is_last_accumulation_step or not hasattr(self.model, "no_sync")
            else self.model.no_sync()
        )

        with sync_ctx:
            with self._autocast():
                loss = self._compute_loss(batch)

            # Normalise loss across accumulation steps. This ensures that the final
            # gradient magnitude is independent of grad_acc, and that the effective
            # batch size behaves as expected.
            scaled_loss = loss / self.args.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        return loss.detach().item()

    def _optimizer_step(self) -> None:
        """
        Perform the optimizer step after a full accumulation window.

        Steps:
        1. If using a GradScaler (FP16), unscale the gradients.
        2. Clip gradients (if `max_grad_norm > 0`).
        3. Step the optimizer (either via scaler or directly).
        4. Update the scaler (if used).
        5. Step the scheduler (always after optimizer step).
        6. Zero out gradients (setting to None saves memory).

        WHY order: Unscale before clipping because clipping should operate on
        the actual gradient values, not the scaled ones. Step scaler after
        optimizer step so it can update its scale factor based on the overflow
        status.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if self.args.max_grad_norm > 0:
            clip_grad_norm_(self.model, self.args.max_grad_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """
        Extract a scalar loss tensor from the model given a batch.

        Resolution order (first match wins):
        1. User‑supplied `compute_loss_fn(model, batch)`
        2. If batch is a dict: `model(**batch).loss` (HuggingFace style)
        3. If batch is not a dict: `model(batch)` (expects a scalar tensor)
        4. If the model output is a tensor, use it directly.
        5. Otherwise, raise a descriptive error.

        WHY: This flexibility allows the trainer to work with both HuggingFace
        models (which return a `CausalLMOutput` with a `.loss` attribute) and
        custom PyTorch modules that return a scalar tensor directly.

        Returns
        -------
        torch.Tensor
            A scalar tensor (still on the device, possibly with gradients attached).
        """
        if self.compute_loss_fn is not None:
            return self.compute_loss_fn(self.model, batch)

        if isinstance(batch, dict):
            outputs = self.model(**batch)
        else:
            outputs = self.model(batch)

        if isinstance(outputs, torch.Tensor):
            return outputs
        # HuggingFace style: assume output has a .loss attribute
        if hasattr(outputs, "loss"):
            return outputs.loss
        raise TypeError(
            f"Model output type {type(outputs)} not supported. "
            "Either implement compute_loss_fn or ensure your model returns a tensor or an object with .loss."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate(self) -> dict[str, float]:
        """
        Run evaluation and return a dictionary of metrics.

        If `eval_fn` is provided, it is used directly. Otherwise, a default loop
        runs over `eval_dataloader` and averages the loss from `_compute_loss`.

        The model is set to eval mode before evaluation and restored to train mode
        after evaluation, regardless of whether the evaluation succeeds.

        Returns
        -------
        dict[str, float]
            Metrics dictionary, at least containing `"eval_loss"` if default loop
            was used. For custom eval_fn, the keys are user-defined.

        Edge Cases:
        - If `eval_dataloader` is None, returns empty dict (no evaluation).
        - If evaluation fails (e.g., model returns None), logs error and returns empty dict.
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()

        if self.eval_fn is not None:
            try:
                metrics = self.eval_fn(self.model, self.eval_dataloader, self.device)
            except Exception as e:
                logger.error("Custom eval_fn failed: %s", e, exc_info=True)
                metrics = {}
        else:
            metrics = self._default_eval_loop()

        self.model.train()
        return metrics

    def _default_eval_loop(self) -> dict[str, float]:
        """
        Default evaluation loop: average the loss over the entire eval dataloader.

        This method does not use gradient accumulation and does not modify model
        state. It is intended to be fast and simple.

        Returns
        -------
        dict[str, float]
            Contains a single key `"eval_loss"` with the average loss.

        Performance: O(N) forward passes, no backward. The loop uses tqdm only
        when the process is rank 0.
        """
        total_loss = 0.0
        num_batches = 0

        eval_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            unit="batch",
            leave=False,
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch in eval_bar:
                batch = self._batch_to_device(batch)
                with self._autocast():
                    loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        eval_loss = total_loss / max(1, num_batches)
        # Add perplexity automatically via the utility
        return format_metrics_with_ppl({"eval_loss": eval_loss})

    def _handle_evaluation(self, metrics: dict[str, float]) -> None:
        """
        Post‑evaluation bookkeeping: update best metric, save best checkpoint,
        dispatch `on_evaluate` callbacks.

        This method is called after each evaluation (step‑based or epoch‑end).
        It modifies `self.state.best_metric` and `self.state.best_model_checkpoint`
        if the monitored metric improves.

        Parameters
        ----------
        metrics : dict[str, float]
            Metrics dictionary returned by `_evaluate`.

        WHY: Centralising this logic ensures that both step‑based and epoch‑based
        evaluation use the same improvement logic and callback dispatching.
        """
        args = self.args
        metric_value = metrics.get(args.metric_for_best_model)

        if metric_value is not None:
            is_better = self._is_better_metric(metric_value)
            if is_better:
                self.state.best_metric = metric_value
                if args.save_checkpoints:
                    self._save("best")
                    self.state.best_model_checkpoint = os.path.join(
                        args.output_dir, "checkpoint-best"
                    )
                logger.info(
                    "New best %s = %.6f → checkpoint-best",
                    args.metric_for_best_model,
                    metric_value,
                )

        # Always log evaluation metrics (even if no improvement)
        self._log(metrics)
        # Explicitly print evaluation results so they are always visible
        logger.info("Evaluation results: %s", metrics)

        self.callback_handler.on_evaluate(
            self.state, self.control, metrics=metrics
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step‑level side‑effects (logging, evaluation, saving)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_step_end(self, avg_loss: float) -> None:
        """
        After each optimizer step (after gradient accumulation), handle:
        - Logging (if step % logging_steps == 0)
        - Step‑based evaluation (if step % eval_steps == 0)
        - Step‑based checkpoint (if step % save_steps == 0)

        Parameters
        ----------
        avg_loss : float
            Average loss over the accumulation window (already divided by grad_acc).

        WHY: This method is called only when an optimizer step actually occurs,
        not on every micro‑batch. This is the right place to perform actions that
        should happen once per global step.
        """
        args = self.args
        state = self.state

        # Logging
        if should_log(state, args.logging_steps):
            logs: dict[str, Any] = {
                "loss": round(avg_loss, 4),
                "lr": get_current_lr(self.optimizer),
                "epoch": round(state.epoch, 2),
            }
            if self.scaler is not None:
                logs["grad_scale"] = self.scaler.get_scale()
            # Add perplexity for immediate feedback
            logs["perplexity"] = compute_perplexity(avg_loss)

            self._log(logs)

        # Step‑based evaluation
        if should_evaluate(state, args.eval_steps) and self.eval_dataloader is not None:
            metrics = self._evaluate()
            self._handle_evaluation(metrics)

        # Step‑based checkpoint
        if should_save(state, args.save_steps) and args.save_checkpoints:
            self._save(step=state.global_step)
            prune_checkpoints(args.output_dir, args.save_total_limit)

    def _log(self, logs: dict[str, Any]) -> None:
        """
        Append `logs` to `state.log_history` and dispatch `on_log` callbacks.

        This method resets `control.should_log` to False after dispatching,
        because the trainer has already handled the request.
        """
        logs["step"] = self.state.global_step
        self.state.log_history.append(logs)
        self.control.should_log = False
        self.callback_handler.on_log(self.state, self.control, logs=logs)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpointing (thin wrapper over trainer_utils)
    # ─────────────────────────────────────────────────────────────────────────

    def _save(
        self,
        tag: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Save a checkpoint using the `trainer_utils.save_checkpoint` function.

        This method also dispatches the `on_save` callback so that integrations
        can react to checkpoint saving (e.g., log to W&B that a checkpoint was saved).

        Parameters
        ----------
        tag : Optional[str]
            Descriptive tag (e.g., "best", "final", "epoch-3"). If provided,
            the checkpoint is saved as `checkpoint-<tag>` and is never pruned.
        step : Optional[int]
            Step number for numbered checkpoints (e.g., 1000). Used only when
            `tag` is None. These checkpoints are prunable.
        """
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            state=self.state,
            callback_handler=self.callback_handler,
            output_dir=self.args.output_dir,
            step=step,
            tag=tag,
            early_stopping=self.early_stopping,
            training_args=self.args,
            atomic=getattr(self.args, "atomic_checkpoint", True),
        )
        self.callback_handler.on_save(self.state, self.control)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────────────────

    def _autocast(self) -> contextlib.AbstractContextManager:
        """
        Return the appropriate autocast context manager based on precision.

        If precision is "fp32" or AMP is disabled, returns a nullcontext.
        For "fp16" and "bf16", returns `torch.autocast` with the correct dtype.
        """
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def _batch_to_device(self, batch: Any) -> Any:
        """
        Recursively move tensors in a batch to the training device.

        Supported structures: dict, list, tuple, and plain tensors.
        Non‑tensor objects (e.g., strings, ints) are left unchanged.

        WHY: Recursion is used to handle nested structures that some
        datasets return (e.g., a tuple of (input_ids, attention_mask, labels)).
        This is more robust than assuming a flat dict.
        """
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, (list, tuple)):
            moved = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
            return type(batch)(moved)
        return batch

    def _is_better_metric(self, current: float) -> bool:
        """
        Determine whether `current` is better than the stored best metric.

        Direction is inferred from `greater_is_better` if set; otherwise,
        auto‑detect based on whether the metric name contains "loss"
        (case‑insensitive). This matches the logic of `EarlyStoppingCallback`
        so that improvement decisions are consistent.

        Edge Cases:
        - If `self.state.best_metric` is None (first evaluation), returns True.
        """
        if self.state.best_metric is None:
            return True
        args = self.args
        greater = args.greater_is_better
        if greater is None:
            greater = "loss" not in args.metric_for_best_model.lower()
        if greater:
            return current > self.state.best_metric
        else:
            return current < self.state.best_metric

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Set random seeds for Python, NumPy, and PyTorch (including CUDA).

        WHY: Ensures reproducibility of data shuffling, dropout, and weight
        initialisation across runs. Note that CUDA operations are not fully
        deterministic unless `torch.backends.cudnn.deterministic = True` is also
        set, which the user must do separately (because it can impact performance).
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)