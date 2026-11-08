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
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kilat.data.dataloader import set_dataloader_epoch
from .args import TrainingArguments
from .callbacks import (
    CallbackHandler,
    EarlyStoppingCallback,
    TrainerControl,
    TrainerState,
)
from .checkpoint import (
    capture_rng_state,
    clip_grad_norm_,
    compute_perplexity,
    compute_total_steps,
    format_metrics_with_ppl,
    get_current_lr,
    get_device,
    get_global_rank,
    get_latest_checkpoint,
    is_distributed_initialized,
    load_checkpoint,
    prune_checkpoints,
    restore_rng_state,
    save_checkpoint,
    should_evaluate,
    should_log,
    should_save,
)
from .integrations import DEFAULT_CALLBACKS, get_reporting_integration_callbacks
from .optimizer import create_optimizer, resolve_amp_dtype
from .scheduler import get_scheduler
from kilat.utils.report import count_parameters

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
        >>> from kilat.engine import KilatTrainer, TrainingArguments
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
        tokenizer: Optional[Any] = None,         
        tokenizer_config: Optional[Any] = None,  
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
            default `ConsoleCallback`.
        tokenizer : Optional[Any]
            Tokenizer instance for saving checkpoints. If provided, the tokenizer
            will be saved alongside the model checkpoint.
        tokenizer_config : Optional[Any]
            Tokenizer configuration (e.g., from MainConfig.tokenizer) used to
            determine if the tokenizer needs to be saved (local vs Hub).
        """
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn
        self.compute_loss_fn = compute_loss_fn
        self.tokenizer = tokenizer                    
        self.tokenizer_config = tokenizer_config      

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
            is_world_process_zero=(get_global_rank() == 0),
        )
        self.control = TrainerControl()
        self._skipped_steps_total: int = 0
        self._skipped_steps_consecutive: int = 0
        self._last_grad_norm: Optional[torch.Tensor] = None
        self._last_ce_loss: Optional[float] = None
        self._last_aux_loss: Optional[float] = None
        self._last_batch: Any = None
        self._last_step_tokens: int = 0
        self._epoch_iter_rng_state: Optional[dict[str, Any]] = None
        self._batches_in_current_cycle: int = 0

        # ─── Callback setup ──────────────────────────────────────────────
        # Order matters: integrations first (so logging captures the start),
        # then early stopping, then user callbacks, and finally the mandatory
        # progress callback (which always logs to console).
        integration_cbs = get_reporting_integration_callbacks(args.report_to)
        early_stopping_cb: Optional[EarlyStoppingCallback] = None
        if args.early_stopping_patience is not None and args.early_stopping_patience >= 1:
            early_stopping_cb = EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
                metric_for_best_model=args.metric_for_best_model,
                greater_is_better=args.greater_is_better,
            )
        self.early_stopping = early_stopping_cb

        all_callbacks = (
            integration_cbs
            + ([early_stopping_cb] if early_stopping_cb is not None else [])
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
                    tokenizer_config=self.tokenizer_config, 
                )
                # Keep the newly computed step budget from the current args.
                # load_checkpoint restores TrainerState from disk, including the
                # old max_steps, but resume should follow the current run config.
                self.state.max_steps = total_steps
                # If load_checkpoint returned a new model, we replace it.
                # Otherwise, the original model was modified in place.

        total_params = count_parameters(self.model)
        trainable_params = count_parameters(self.model, trainable_only=True)
        logger.info(
            "Model parameters | total=%s | trainable=%s",
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

        # ─── Notify callbacks that training is about to start ─────────────
        self.callback_handler.on_train_begin(
            self.state, self.control, model=self.model, trainer=self
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
                if "ppl" in metrics:
                    postfix["ppl"] = f"{metrics['ppl']:.2f}"
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
        acc_ce_loss = 0.0
        acc_aux_loss = 0.0
        acc_steps = 0
        grad_acc = args.gradient_accumulation_steps

        resumed_dl = getattr(self.state, "_resumed_dataloader_state", None)
        cycle_epoch = int(self.state.epoch)
        set_dataloader_epoch(self.train_dataloader, cycle_epoch)

        if isinstance(resumed_dl, dict) and resumed_dl.get("epoch_iter_rng_state") is not None:
            # Deterministic exact-batch fast-forward to preserve bitwise RNG + sampler parity
            post_step_rng = capture_rng_state()
            self._epoch_iter_rng_state = resumed_dl["epoch_iter_rng_state"]
            restore_rng_state(self._epoch_iter_rng_state)
            data_iter = iter(self.train_dataloader)
            skip_batches = int(resumed_dl.get("batches_in_current_cycle", 0))
            for _ in range(skip_batches):
                try:
                    next(data_iter)
                except StopIteration:
                    break
            self._batches_in_current_cycle = skip_batches
            restore_rng_state(post_step_rng)
            object.__setattr__(self.state, "_resumed_dataloader_state", None)
        else:
            self._epoch_iter_rng_state = capture_rng_state()
            self._batches_in_current_cycle = 0
            data_iter = iter(self.train_dataloader)

        self.callback_handler.on_epoch_begin(self.state, self.control)

        while self.state.global_step < total_steps:
            try:
                batch = next(data_iter)
                self._batches_in_current_cycle += 1
            except StopIteration:
                self.callback_handler.on_epoch_end(self.state, self.control)
                self.state.epoch += 1.0
                cycle_epoch = int(self.state.epoch)
                set_dataloader_epoch(self.train_dataloader, cycle_epoch)
                self.callback_handler.on_epoch_begin(self.state, self.control)
                self._epoch_iter_rng_state = capture_rng_state()
                self._batches_in_current_cycle = 0
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
                self._batches_in_current_cycle = 1

            self.callback_handler.on_step_begin(self.state, self.control)

            loss = self._forward_backward(
                batch,
                is_last_accumulation_step=(acc_steps + 1) >= grad_acc,
                accum_divisor=grad_acc,
            )
            acc_loss += loss
            acc_ce_loss += self._last_ce_loss if self._last_ce_loss is not None else loss
            acc_aux_loss += self._last_aux_loss if self._last_aux_loss is not None else 0.0
            acc_steps += 1

            self.callback_handler.on_substep_end(self.state, self.control)

            if acc_steps >= grad_acc:
                self._optimizer_step()
                self.state.global_step += 1
                avg_loss = acc_loss / grad_acc
                avg_ce_loss = acc_ce_loss / grad_acc
                avg_aux_loss = acc_aux_loss / grad_acc
                acc_loss = 0.0
                acc_ce_loss = 0.0
                acc_aux_loss = 0.0
                acc_steps = 0

                step_bar.update(1)
                step_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ppl=f"{compute_perplexity(avg_ce_loss):.2f}",
                    lr=f"{get_current_lr(self.optimizer):.2e}",
                )

                self.callback_handler.on_step_end(
                    self.state,
                    self.control,
                    model=self.model,
                    trainer=self,
                    avg_loss=avg_loss,
                    avg_ce_loss=avg_ce_loss,
                    avg_aux_loss=avg_aux_loss,
                )
                self._handle_step_end(avg_loss, avg_ce_loss=avg_ce_loss, avg_aux_loss=avg_aux_loss)
                self._last_step_tokens = 0

                if self.control.should_training_stop:
                    break

        step_bar.close()

    def _run_epoch(self, epoch: int, num_epochs: int) -> None:
        """
        Run one full pass over the training DataLoader with DDP epoch-shuffling
        and exact mid-epoch resumption support.
        """
        args = self.args
        self.model.train()
        set_dataloader_epoch(self.train_dataloader, epoch)

        steps_in_epoch = len(self.train_dataloader)
        grad_acc = args.gradient_accumulation_steps
        denom = max(1, steps_in_epoch)

        resumed_dl = getattr(self.state, "_resumed_dataloader_state", None)
        skip_batches = 0
        if isinstance(resumed_dl, dict) and resumed_dl.get("epoch_iter_rng_state") is not None:
            post_step_rng = capture_rng_state()
            self._epoch_iter_rng_state = resumed_dl["epoch_iter_rng_state"]
            restore_rng_state(self._epoch_iter_rng_state)
            raw_iter = iter(self.train_dataloader)
            skip_batches = int(resumed_dl.get("batches_in_current_cycle", 0))
            for _ in range(skip_batches):
                try:
                    next(raw_iter)
                except StopIteration:
                    break
            self._batches_in_current_cycle = skip_batches
            restore_rng_state(post_step_rng)
            object.__setattr__(self.state, "_resumed_dataloader_state", None)
        else:
            self._epoch_iter_rng_state = capture_rng_state()
            self._batches_in_current_cycle = 0
            raw_iter = iter(self.train_dataloader)

        step_bar = tqdm(
            total=steps_in_epoch,
            initial=skip_batches,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
            leave=False,
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        acc_loss = 0.0
        acc_ce_loss = 0.0
        acc_aux_loss = 0.0
        acc_steps = 0
        window_size = grad_acc

        for local_step in range(skip_batches, steps_in_epoch):
            try:
                batch = next(raw_iter)
            except StopIteration:
                break
            self._batches_in_current_cycle = local_step + 1
            step_bar.update(1)

            if acc_steps == 0:
                window_size = max(1, min(grad_acc, steps_in_epoch - local_step))

            self.state.epoch = float(epoch) + float(local_step + 1) / float(denom)

            is_last_acc_step = (
                (local_step + 1) % grad_acc == 0
                or (local_step + 1) == steps_in_epoch
            )

            self.callback_handler.on_step_begin(self.state, self.control)

            loss = self._forward_backward(
                batch,
                is_last_accumulation_step=is_last_acc_step,
                accum_divisor=window_size,
            )
            acc_loss += loss
            acc_ce_loss += self._last_ce_loss if self._last_ce_loss is not None else loss
            acc_aux_loss += self._last_aux_loss if self._last_aux_loss is not None else 0.0
            acc_steps += 1

            self.callback_handler.on_substep_end(self.state, self.control)

            if is_last_acc_step:
                self._optimizer_step()
                self.state.global_step += 1
                avg_loss = acc_loss / acc_steps
                avg_ce_loss = acc_ce_loss / acc_steps
                avg_aux_loss = acc_aux_loss / acc_steps
                acc_loss = 0.0
                acc_ce_loss = 0.0
                acc_aux_loss = 0.0
                acc_steps = 0

                step_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ppl=f"{compute_perplexity(avg_ce_loss):.2f}",
                    lr=f"{get_current_lr(self.optimizer):.2e}",
                )

                self.callback_handler.on_step_end(
                    self.state,
                    self.control,
                    model=self.model,
                    trainer=self,
                    avg_loss=avg_loss,
                    avg_ce_loss=avg_ce_loss,
                    avg_aux_loss=avg_aux_loss,
                )
                self._handle_step_end(avg_loss, avg_ce_loss=avg_ce_loss, avg_aux_loss=avg_aux_loss)
                self._last_step_tokens = 0

                if self.control.should_training_stop or self.control.should_epoch_stop:
                    break

        step_bar.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Core compute primitives
    # ─────────────────────────────────────────────────────────────────────────

    def _unwrap_model(self) -> nn.Module:
        raw = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
        return raw

    def _forward_backward(
        self,
        batch: Any,
        is_last_accumulation_step: bool,
        accum_divisor: Optional[int] = None,
    ) -> float:
        """
        Run one forward + backward pass for a single micro‑batch.
        """
        batch = self._batch_to_device(batch)
        self._last_batch = batch
        if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor):
            self._last_step_tokens += int(batch["input_ids"].numel())
        elif isinstance(batch, torch.Tensor):
            self._last_step_tokens += int(batch.numel())
        divisor = max(1, accum_divisor if accum_divisor is not None else self.args.gradient_accumulation_steps)

        sync_ctx = (
            contextlib.nullcontext()
            if is_last_accumulation_step or not hasattr(self.model, "no_sync")
            else self.model.no_sync()
        )

        with sync_ctx:
            with self._autocast():
                loss = self._compute_loss(batch)

            scaled_loss = loss / float(divisor)

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        raw_model = self._unwrap_model()
        breakdown = getattr(raw_model, "_last_loss_breakdown", None) if self.compute_loss_fn is None else None
        if isinstance(breakdown, dict) and "ce_loss" in breakdown:
            self._last_ce_loss = float(breakdown["ce_loss"].item())
            self._last_aux_loss = float(breakdown["aux_loss"].item()) if "aux_loss" in breakdown else 0.0
        else:
            self._last_ce_loss = None
            self._last_aux_loss = None

        return loss.detach().item()

    def _optimizer_step(self) -> None:
        """
        Perform the optimizer step after a full accumulation window.
        Guards lr_scheduler.step() when GradScaler skips an Inf/NaN step.
        """
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if self.args.max_grad_norm > 0:
            self._last_grad_norm = clip_grad_norm_(self.model, self.args.max_grad_norm).detach()
        else:
            self._last_grad_norm = None

        if self.scaler is not None:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_was_run = self.scaler.get_scale() >= scale_before
        else:
            self.optimizer.step()
            optimizer_was_run = True

        if optimizer_was_run:
            self.scheduler.step()
            self._skipped_steps_consecutive = 0
            raw_model = self._unwrap_model()
            for mod in raw_model.modules():
                if hasattr(mod, "update_expert_bias"):
                    mod.update_expert_bias()
        else:
            self._skipped_steps_total += 1
            self._skipped_steps_consecutive += 1

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
        Default evaluation loop computing exact token-weighted pure cross-entropy NLL
        so perplexity exp(eval_loss) is never polluted by MoE auxiliary losses or
        skewed by shorter/padded trailing batches.
        """
        total_loss = 0.0
        num_batches = 0
        total_nll_sum = 0.0
        total_valid_tokens = 0
        total_aux_sum = 0.0
        used_token_weighted = False

        eval_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            unit="batch",
            leave=False,
            disable=not self.state.is_world_process_zero,
            dynamic_ncols=True,
        )

        raw_model = self._unwrap_model()

        saved_eval_rng = capture_rng_state()
        try:
            with torch.no_grad():
                for batch in eval_bar:
                    batch = self._batch_to_device(batch)
                    with self._autocast():
                        loss = self._compute_loss(batch)
                    breakdown = (
                        getattr(raw_model, "_last_loss_breakdown", None)
                        if self.compute_loss_fn is None
                        else None
                    )
                    if isinstance(breakdown, dict) and "nll_sum" in breakdown and "valid_tokens" in breakdown:
                        used_token_weighted = True
                        total_nll_sum += float(breakdown["nll_sum"].item())
                        total_valid_tokens += int(breakdown["valid_tokens"].item())
                        total_aux_sum += float(breakdown.get("aux_loss", torch.tensor(0.0)).item())
                    total_loss += loss.item()
                    num_batches += 1
        finally:
            restore_rng_state(saved_eval_rng)

        if used_token_weighted:
            if is_distributed_initialized():
                stats = torch.tensor(
                    [total_nll_sum, float(total_valid_tokens), total_aux_sum, float(num_batches)],
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                total_nll_sum = float(stats[0].item())
                total_valid_tokens = int(stats[1].item())
                total_aux_sum = float(stats[2].item())
                num_batches = int(stats[3].item())
            eval_loss = total_nll_sum / float(max(1, total_valid_tokens))
            base_metrics: dict[str, float] = {"eval_loss": eval_loss}
            if total_aux_sum > 0.0 and num_batches > 0:
                base_metrics["eval_aux_loss"] = total_aux_sum / float(num_batches)
            return format_metrics_with_ppl(base_metrics)

        if is_distributed_initialized():
            fallback_stats = torch.tensor([total_loss, float(num_batches)], dtype=torch.float64, device=self.device)
            dist.all_reduce(fallback_stats, op=dist.ReduceOp.SUM)
            total_loss = float(fallback_stats[0].item())
            num_batches = int(fallback_stats[1].item())
        eval_loss = total_loss / max(1, num_batches)
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

        self.callback_handler.on_evaluate(
            self.state, self.control, metrics=metrics
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step‑level side‑effects (logging, evaluation, saving)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_step_end(
        self,
        avg_loss: float,
        avg_ce_loss: Optional[float] = None,
        avg_aux_loss: Optional[float] = None,
    ) -> None:
        """
        After each optimizer step (after gradient accumulation), handle:
        - Logging (if step % logging_steps == 0)
        - Step‑based evaluation (if step % eval_steps == 0)
        - Step‑based checkpoint (if step % save_steps == 0)
        """
        args = self.args
        state = self.state
        pure_ce = avg_ce_loss if avg_ce_loss is not None else avg_loss

        # Logging
        if should_log(state, args.logging_steps):
            logs: dict[str, Any] = {
                "loss": round(avg_loss, 4),
                "ce_loss": round(pure_ce, 4),
                "lr": get_current_lr(self.optimizer),
                "epoch": round(state.epoch, 2),
            }
            if avg_aux_loss is not None and avg_aux_loss > 0.0:
                logs["aux_loss"] = round(avg_aux_loss, 6)
            if self.scaler is not None:
                logs["grad_scale"] = self.scaler.get_scale()
            # Perplexity is strictly exp(pure CE loss), never polluted by MoE aux_loss
            logs["perplexity"] = compute_perplexity(pure_ce)
            logs["ppl"] = logs["perplexity"]

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
        self.callback_handler.on_log(self.state, self.control, logs=logs, model=self.model, trainer=self)

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
        dl_state = {
            "epoch_iter_rng_state": self._epoch_iter_rng_state,
            "batches_in_current_cycle": self._batches_in_current_cycle,
        }
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
            additional_state={"dataloader_state": dl_state},
            training_args=self.args,
            tokenizer=getattr(self, "tokenizer", None),
            tokenizer_config=getattr(self, "tokenizer_config", None),
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