from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from distiliation.losses import BaseDistillLoss, build_loss
from utils.callback import CallbackHandler, EarlyStoppingCallback
from training.arguments import TrainingArguments
from training.checkpointing import prune_checkpoints, resume_from_checkpoint, save_checkpoint
from training.logging_utils import (
    finish_wandb,
    init_wandb,
    log_eval_summary,
    log_final_summary,
    log_training_metrics,
    print_training_header,
)
from training.optim_utils import (
    compute_total_steps,
    create_optimizer,
    create_scheduler,
    resolve_amp_dtype,
)


def _iterable_from_dataset(ds: Dataset) -> IterableDataset:
    """Wrap a map‑style Dataset as an IterableDataset for streaming DataLoader."""
    class _Wrapper(IterableDataset):
        def __init__(self, inner: Dataset):
            self._inner = inner

        def __iter__(self):
            return iter(self._inner)

    return _Wrapper(ds)


class DistillTrainer:
    """
    Trainer for knowledge distillation.

    WHY: Distillation has specific needs: two models (student + teacher),
    two loss components (KL + CE), and teacher must stay frozen.
    Reusing the base training utilities (optimizer, scheduler, checkpointing)
    while adding distillation‑specific logic.

    Key design decisions:
    - Teacher is always in eval mode, gradients disabled.
    - Loss function is pluggable (vanilla KD, reverse KL, adaptive temp).
    - Streaming datasets are supported by using batch_size=None and a wrapper.
    - The student and teacher vocab sizes are validated at init to avoid shape mismatches.
    - Checkpointing saves the student only (teacher is assumed recoverable separately).
    - Early stopping is based on evaluation loss (optional).

    Trade‑offs:
    - We do NOT support gradient checkpointing or activation offloading for teacher.
    - For large teachers, the forward pass may dominate VRAM; user should use bf16.
    - The trainer does not handle tokenizer alignment automatically.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
        loss_fn: Optional[BaseDistillLoss] = None,
        loss_name: str = "vanilla",
        loss_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the distillation trainer.

        Args:
            student: Trainable student model.
            teacher: Frozen teacher model (must provide logits).
            args: Training configuration (batch size, steps, precision, etc.).
            train_dataset: Training data.
            eval_dataset: Optional evaluation data for early stopping / metrics.
            data_collator: Collates raw samples into batches. If None and dataset
                           is map‑style, default_collate is used.
            loss_fn: Explicit loss instance. If None, built from loss_name + kwargs.
            loss_name: Key in loss registry (e.g., "vanilla", "reverse", "adaptive").
            loss_kwargs: Passed to loss constructor.

        Raises:
            ValueError: If student and teacher vocab sizes differ.
        """
        self.student = student
        self.teacher = teacher
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.loss_fn = loss_fn or build_loss(loss_name, **(loss_kwargs or {}))

        # We'll checkpoint the student (or its underlying model if wrapped)
        self._checkpoint_model = getattr(self.student, "model", self.student)
        self._amp_dtype: Optional[torch.dtype] = resolve_amp_dtype(args.precision)
        self._amp_enabled: bool = self._amp_dtype is not None
        self._scaler_enabled: bool = args.precision == "fp16"
        self._autocast_device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Reproducibility
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)
        # Teacher never trains; we enforce eval mode and disable gradients.
        self.teacher.eval()
        self.loss_fn.to(self.device)

        # ---------- DataLoader construction for streaming vs map-style ----------
        # WHY: Streaming datasets (e.g., from web) cannot be shuffled or indexed.
        # We use a wrapper that implements IterableDataset and set batch_size=None
        # so the DataLoader yields individual samples. The collation happens inside
        # _load_batch (which handles both dicts and tuples).
        is_streaming = getattr(self.train_dataset, "streaming", False)
        self._train_is_streaming = is_streaming
        if is_streaming:
            train_loader_dataset = _iterable_from_dataset(self.train_dataset)
            self.train_dataloader = DataLoader(
                train_loader_dataset,
                batch_size=None,
                shuffle=False,
                collate_fn=None,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                pin_memory=torch.cuda.is_available(),
            )

        if self.eval_dataset is not None:
            self._eval_is_streaming = getattr(self.eval_dataset, "streaming", False)
            if self._eval_is_streaming:
                eval_loader_dataset = _iterable_from_dataset(self.eval_dataset)
                self.eval_dataloader: Optional[DataLoader] = DataLoader(
                    eval_loader_dataset,
                    batch_size=None,
                    shuffle=False,
                    collate_fn=None,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                self.eval_dataloader = DataLoader(
                    self.eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    shuffle=False,
                    collate_fn=self.data_collator,
                    pin_memory=torch.cuda.is_available(),
                )
            self.early_stopping: Optional[EarlyStoppingCallback] = EarlyStoppingCallback(
                patience=args.early_stopping_patience,
                threshold=args.early_stopping_threshold,
            )
        else:
            self.eval_dataloader = None
            self._eval_is_streaming = False
            self.early_stopping = None

        self.callbacks = CallbackHandler(
            [self.early_stopping] if self.early_stopping is not None else []
        )

        # ---------- Vocab size validation ----------
        # WHY: KL divergence requires matching vocabulary dimensions. Mismatch
        # would cause runtime shape error. We fail early.
        if hasattr(self.teacher, "vocab_size") and hasattr(self.student, "vocab_size"):
            if int(self.teacher.vocab_size) != int(self.student.vocab_size):
                raise ValueError(
                    "student and teacher vocab sizes must match for distillation "
                    f"(student={self.student.vocab_size}, teacher={self.teacher.vocab_size})."
                )

        # ---------- Optimizer & scheduler ----------
        # We bundle student + loss_fn because the loss may have learnable parameters
        # (e.g., AdaptiveKDLoss has a learnable temperature). This ensures they
        # all get optimized.
        bundle = nn.Module()
        bundle.student = self.student
        bundle.loss_fn = self.loss_fn
        self._optimizer_bundle = bundle

        if self._train_is_streaming:
            # For streaming, we cannot know the exact number of batches per epoch.
            # We use len(dataset) if available, else fallback to 1 (for step‑mode training).
            dataloader_len = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 1
        else:
            dataloader_len = len(self.train_dataloader)

        self.total_steps = compute_total_steps(
            args.training_mode,
            args.max_steps,
            args.num_train_epochs,
            dataloader_len,
            args.gradient_accumulation_steps,
        )

        self.optimizer = create_optimizer(
            self._optimizer_bundle,
            args.learning_rate,
            args.weight_decay,
        )
        self.scheduler = create_scheduler(self.optimizer, self.total_steps, args.warmup_steps)
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self._scaler_enabled)

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_eval_loss: float = float("inf")
        self.start_time: float = time.time()

        # ---------- Resume from checkpoint ----------
        if args.resume_from_checkpoint is not None:
            self.global_step, self.current_epoch, self.best_eval_loss = resume_from_checkpoint(
                self._checkpoint_model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.early_stopping,
                args.resume_from_checkpoint,
                self.device,
            )
            self._load_loss_state(args.resume_from_checkpoint)

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
            getattr(self._checkpoint_model.config, "to_dict", lambda: {})(),
        )

    # ---------- Helper methods for batch handling and loss state ----------
    def _load_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract input_ids, labels, and optional attention_mask from a batch.

        WHY: Different dataset formats exist:
        - Tuple/list: (input_ids, labels, [attention_mask])
        - Dict: {"input_ids": ..., "labels": ..., "attention_mask": ...}
        - Streaming: may return a single sample (batch_size=None) → need to add batch dim.

        We also handle 3D inputs (e.g., when batch_size=None and DataLoader wraps a single
        sample into a tensor) by squeezing the extra dimension.
        """
        attention_mask: Optional[torch.Tensor] = None
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch[:2]
            if len(batch) > 2:
                attention_mask = batch[2]
            # If batch comes from streaming with batch_size=None, DataLoader may wrap
            # a sample into a 1‑element tensor → we need to remove the batch dim.
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 3:
                input_ids = input_ids.squeeze(0)
                labels = labels.squeeze(0)
                if attention_mask is not None and attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(0)
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")

        # Ensure at least 2D [batch, seq_len]
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        return input_ids, labels, attention_mask

    def _loss_state_path(self, checkpoint_path: str) -> str:
        return os.path.join(checkpoint_path, "distill_state.pt")

    def _save_loss_state(self, checkpoint_path: str) -> None:
        """Save loss function state (e.g., learned temperature of AdaptiveKDLoss)."""
        torch.save({"loss_fn": self.loss_fn.state_dict()}, self._loss_state_path(checkpoint_path))

    def _load_loss_state(self, checkpoint_path: str) -> None:
        """Restore loss function state from checkpoint."""
        loss_state_path = self._loss_state_path(checkpoint_path)
        if os.path.exists(loss_state_path):
            state = torch.load(loss_state_path, map_location=self.device, weights_only=True)
            if "loss_fn" in state:
                self.loss_fn.load_state_dict(state["loss_fn"])

    # ---------- Main training loop ----------
    def train(self) -> None:
        """Entry point: runs either step‑based or epoch‑based training."""
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
        self.student.train()
        self.teacher.eval()
        self.callbacks.on_train_begin(self)

        try:
            if self.args.training_mode == "steps":
                self._train_by_steps()
            else:
                self._train_by_epochs()
        except KeyboardInterrupt:
            print(f"\n{'=' * 60}")
            print(f"Training interrupted by user at step {self.global_step}")
            print("Saving checkpoint before exit...")
            print(f"{'=' * 60}")
            self._save_checkpoint(self.global_step, tag="interrupted")
            self._finish()
            raise

    def _train_by_steps(self) -> None:
        """Training with a fixed number of steps (max_steps)."""
        progress_bar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            desc="Distill (steps)",
            dynamic_ncols=True,
            unit="step",
        )

        step_within_accum = 0
        running_loss = 0.0
        epoch = self.current_epoch or 1
        while self.global_step < self.total_steps:
            self.current_epoch = epoch
            for batch in self.train_dataloader:
                loss_val = self._forward_backward(batch)
                running_loss += loss_val
                step_within_accum += 1

                if step_within_accum == self.args.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1
                    step_within_accum = 0
                    current_lr = self.scheduler.get_last_lr()[0]
                    avg_loss = running_loss / self.args.gradient_accumulation_steps
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

                    if self.eval_dataloader is not None and self.global_step % self.args.eval_steps == 0:
                        if self._run_eval_and_check_stopping():
                            progress_bar.close()
                            return

                    if self.args.save_checkpoints and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    if self.global_step >= self.total_steps:
                        print(f"\n{'=' * 60}")
                        print(f"Training complete ({self.total_steps:,} steps)")
                        print(f"{'=' * 60}")
                        if self.args.save_checkpoints:
                            self._save_checkpoint(self.global_step, tag="final")
                        progress_bar.close()
                        self._finish()
                        return
            epoch += 1

        progress_bar.close()

    def _train_by_epochs(self) -> None:
        """Training for a fixed number of epochs (num_train_epochs)."""
        start_epoch = self.current_epoch or 1
        for epoch in range(start_epoch, self.args.num_train_epochs + 1):
            self.current_epoch = epoch
            total_batches = len(self.train_dataset) if self._train_is_streaming and hasattr(self.train_dataset, "__len__") else len(self.train_dataloader)
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=total_batches,
                desc=f"Distill epoch {epoch}/{self.args.num_train_epochs}",
                dynamic_ncols=True,
                unit="batch",
                leave=True,
            )

            epoch_loss = 0.0
            step_within_accum = 0

            for batch_idx, batch in progress_bar:
                loss_val = self._forward_backward(batch)
                epoch_loss += loss_val
                step_within_accum += 1

                if step_within_accum == self.args.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1
                    step_within_accum = 0
                    current_lr = self.scheduler.get_last_lr()[0]
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
                        epoch_loss = 0.0

                    if self.eval_dataloader is not None and self.global_step % self.args.eval_steps == 0:
                        if self._run_eval_and_check_stopping():
                            return

                    if self.args.save_checkpoints and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(self.global_step)

            print(f"\n[Epoch {epoch}] Complete.")
            if self.eval_dataloader is not None and self._run_eval_and_check_stopping():
                return
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag=f"epoch-{epoch}")

        print(f"\n{'=' * 60}")
        print(f"Training complete ({self.args.num_train_epochs} epochs, {self.global_step:,} steps)")
        print(f"{'=' * 60}")
        if self.args.save_checkpoints:
            self._save_checkpoint(self.global_step, tag="final")
        self._finish()

    # ---------- Forward / backward utilities ----------
    def _forward_backward(self, batch: Any) -> float:
        """
        Single forward + backward pass.

        Steps:
        1. Extract input_ids, labels, mask.
        2. Move to device.
        3. Autocast context for mixed precision.
        4. Get teacher logits (no grad), student logits.
        5. Compute distillation loss (KL + CE).
        6. Scale loss and backpropagate.

        Returns total loss value (before dividing by accumulation steps) for logging.
        """
        input_ids, labels, attention_mask = self._load_batch(batch)
        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        with torch.amp.autocast(
            device_type=self._autocast_device,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            with torch.no_grad():
                teacher_logits = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            student_logits = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            loss_output = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                attention_mask=attention_mask,
            )
            # Normalise by accumulation steps to keep loss scale consistent.
            loss = loss_output.total_loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        self._last_loss_output = loss_output
        return loss_output.total_loss.item()

    def _optimizer_step(self) -> torch.Tensor:
        """
        Apply gradient clipping, optimizer step, and scheduler step.

        Returns gradient norm for logging.
        """
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self._optimizer_bundle.parameters(),
            max_norm=self.args.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    # ---------- Evaluation & early stopping ----------
    def _run_eval_and_check_stopping(self) -> bool:
        """
        Run evaluation, log results, and check if early stopping is triggered.

        Returns True if training should stop (early stopping triggered).
        """
        eval_loss, eval_ppl = self.evaluate()
        self.student.train()
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag="best")
        if self.callbacks.on_evaluate_end(self, eval_loss, eval_ppl):
            print(f"\n{'=' * 60}")
            print(f"Early stopping triggered at step {self.global_step}")
            print(f"{'=' * 60}")
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag="early-stopped")
            self._finish()
            return True
        return False

    @torch.inference_mode()
    def evaluate(self) -> tuple[float, float]:
        """Compute average evaluation loss and perplexity over the eval dataset."""
        self.student.eval()
        self.teacher.eval()
        eval_loss = 0.0
        eval_progress = tqdm(
            self.eval_dataloader,
            desc="Evaluating distill",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        )

        for batch in eval_progress:
            input_ids, labels, attention_mask = self._load_batch(batch)
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self._autocast_device,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                teacher_logits = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                student_logits = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                loss_output = self.loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                eval_loss += loss_output.total_loss.item()

            eval_progress.set_postfix({"loss": f"{loss_output.total_loss.item():.4f}"})

        # For streaming, we may not know the exact number of batches; fallback to 1.
        if self._eval_is_streaming:
            num_eval_batches = len(self.eval_dataset) if self.eval_dataset is not None and hasattr(self.eval_dataset, "__len__") else 1
        else:
            num_eval_batches = len(self.eval_dataloader)

        avg_eval_loss = eval_loss / max(1, num_eval_batches)
        eval_ppl = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float("inf")

        log_eval_summary(
            avg_eval_loss,
            eval_ppl,
            self.global_step,
            self.best_eval_loss,
            self.args.report_to,
        )
        return avg_eval_loss, eval_ppl

    # ---------- Checkpointing ----------
    def _save_checkpoint(self, step: int, tag: Optional[str] = None) -> None:
        """Save student, optimizer, scheduler, scaler, and early stopping state."""
        if not self.args.save_checkpoints:
            return

        save_checkpoint(
            self._checkpoint_model,
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
        # Also save loss function state (e.g., learnable temperature)
        self._save_loss_state(
            os.path.join(
                self.args.output_dir,
                f"checkpoint-{step}" if tag is None else f"checkpoint-{tag}",
            )
        )
        prune_checkpoints(self.args.output_dir, self.args.save_total_limit)

    def _finish(self) -> None:
        """Final logging and cleanup."""
        log_final_summary(
            self.global_step,
            self.start_time,
            self.best_eval_loss,
            self.args.output_dir,
        )
        self.callbacks.on_train_end(self)
        finish_wandb(self.args.report_to)