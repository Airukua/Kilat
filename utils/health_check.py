from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Optional
import torch
from torch.utils.data import Dataset, IterableDataset, Subset
from arc.model import KilatTransformer
from training.arguments import TrainingArguments
from training.trainer import KilatTrainer


@dataclass(frozen=True)
class HealthCheckReport:
    """
    Immutable result of a training health check that validates checkpointing and resume.

    WHY: Training pipelines often break silently on resume or checkpoint corruption.
    This report provides explicit success/failure and diagnostic info to catch
    such issues before a long training run.
    """
    success: bool
    output_dir: str
    first_run_steps: int
    resumed_steps: int
    checkpoint_path: str
    notes: tuple[str, ...]

    def pretty(self) -> str:
        """Human‑readable report suitable for console logs or CI output."""
        lines = [
            f"Health check: {'PASS' if self.success else 'FAIL'}",
            f"Output dir: {self.output_dir}",
            f"Checkpoint: {self.checkpoint_path}",
            f"First run steps: {self.first_run_steps}",
            f"Resumed steps: {self.resumed_steps}",
        ]
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


class _IterablePrefixDataset(IterableDataset):
    """
    Wraps an iterable dataset and yields at most `limit` items.

    WHY: Streaming datasets (e.g., from web or disk streams) don't support
    slicing or __getitem__. This adapter provides a uniform way to take a
    prefix for health checks without loading the entire dataset.
    """
    def __init__(self, inner: IterableDataset | Dataset, limit: int):
        self.inner = inner
        self.limit = limit
        # Mark as streaming so `_take_first_examples` knows it's not subscriptable
        self.streaming = True

    def __iter__(self):
        count = 0
        for item in self.inner:
            if count >= self.limit:
                break
            count += 1
            yield item

    def __len__(self):
        return self.limit


def _take_first_examples(dataset: Dataset | IterableDataset, limit: int) -> Dataset | IterableDataset:
    """
    Return a view containing at most `limit` examples from the start of the dataset.

    Decision logic (order matters):
    1. If dataset is streaming or lacks __getitem__, wrap with _IterablePrefixDataset.
    2. Else if it has __len__, use torch Subset (efficient, no copy).
    3. Fallback to iterable wrapper (safe but slightly slower).

    WHY: Health checks need only a tiny sample; we must avoid copying large data
    or forcing indexable access on streaming sources.
    """
    if limit <= 0:
        raise ValueError(f"limit must be > 0, got {limit}.")

    # Streaming datasets cannot be subscripted reliably; use iterator wrapper.
    if getattr(dataset, "streaming", False) or not hasattr(dataset, "__getitem__"):
        return _IterablePrefixDataset(dataset, limit)

    # Standard map-style dataset with known length → efficient Subset.
    if hasattr(dataset, "__len__"):
        return Subset(dataset, range(min(limit, len(dataset))))

    # Defensive fallback (e.g., custom dataset without __len__ but with __getitem__)
    return _IterablePrefixDataset(dataset, limit)


def _find_numbered_checkpoint(output_dir: str) -> Path:
    """
    Locate the most recent checkpoint of form `checkpoint-<N>` in output_dir.

    WHY: The trainer creates numbered checkpoints; resume requires the exact path.
    Sorting by integer suffix is safer than by modification time (which can be
    affected by rsync or filesystem quirks).

    Raises FileNotFoundError if no such checkpoint exists.
    """
    checkpoint_dir = Path(output_dir)
    numbered = sorted(
        [
            path
            for path in checkpoint_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.replace("checkpoint-", "", 1).isdigit()
        ],
        # Extract integer suffix for numeric sorting
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    if not numbered:
        raise FileNotFoundError(f"No numbered checkpoint found in {output_dir}.")
    return numbered[-1]


def _clone_training_args(args: TrainingArguments, **overrides: Any) -> TrainingArguments:
    """
    Create a shallow copy of TrainingArguments with overridden attributes.

    WHY: TrainingArguments is a dataclass but not frozen. Direct mutation can
    cause subtle bugs when the same args object is reused. Cloning ensures
    isolation between first run and resume run.
    """
    values = vars(args).copy()
    values.update(overrides)
    return TrainingArguments(**values)


def run_health_check(
    model: KilatTransformer,
    train_dataset: Dataset | IterableDataset,
    *,
    eval_dataset: Dataset | IterableDataset | None = None,
    data_collator: Any | None = None,
    args: TrainingArguments | None = None,
    output_dir: str | None = None,
    sample_count: int = 1,
    first_run_steps: int = 1,
    resumed_target_steps: int = 2,
) -> HealthCheckReport:
    """
    Validate training checkpointing and resume capability on a tiny dataset slice.

    WHAT THIS DOES:
    1. Train for `first_run_steps` steps on `sample_count` examples.
    2. Locate the saved checkpoint.
    3. Create a fresh model instance and resume training from that checkpoint.
    4. Train for `resumed_target_steps` steps (must be > first_run_steps).
    5. Report success if both runs completed and final checkpoint exists.

    WHY THIS IS NEEDED:
    - Checkpoint/resume is a common failure point (missing files, state mismatch,
      optimizer parameter groups, random seed desync).
    - Running this before a long training job prevents wasting GPU hours.
    - Uses a temporary directory by default, so it doesn't pollute actual outputs.

    TRADE‑OFFS:
    - We clone the model config but not the weights to ensure resume loads
      everything from the checkpoint. This catches missing state errors.
    - We force batch size = 1, grad accumulation = 1 to minimize memory and
      make step counting deterministic.
    - The precision is taken from `args.precision` (if available) or falls back
      to fp32 to avoid precision‑related resume issues.

    EDGE CASES:
    - Streaming datasets are handled via _IterablePrefixDataset.
    - If `output_dir` is None, we use a temporary directory (cleaned up by OS
      later, but not automatically deleted – caller may want to inspect).
    - The function assumes the trainer saves a `training_state.pt` inside the
      checkpoint directory (contains optimizer state, step count, etc.).
    """
    if first_run_steps < 1:
        raise ValueError("first_run_steps must be >= 1.")
    if resumed_target_steps <= first_run_steps:
        raise ValueError("resumed_target_steps must be greater than first_run_steps.")

    notes: list[str] = [
        f"Using first {sample_count} sample(s) from the training dataset.",
        "Checkpoint resume is validated with a fresh model instance.",
    ]
    chosen_precision = "fp32"
    if args is not None and torch.cuda.is_available():
        # NOTE: Only use the user's precision if CUDA is available; on CPU fp16/bf16
        # may not be supported and would cause errors.
        chosen_precision = args.precision

    # Use a temp dir by default to avoid cluttering real experiment directories.
    health_output_dir = (
        tempfile.mkdtemp(prefix="kilat_health_")
        if output_dir is None
        else output_dir
    )
    health_train_dataset = _take_first_examples(train_dataset, sample_count)
    health_eval_dataset = (
        _take_first_examples(eval_dataset, sample_count)
        if eval_dataset is not None
        else None
    )

    # Baseline arguments: minimal stepping, single batch, no external logging.
    base_args = args or TrainingArguments(
        output_dir=health_output_dir,
        training_mode="steps",
        max_steps=first_run_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=2,
        save_checkpoints=True,
        report_to="none",
        precision=chosen_precision,
    )

    # First run: train from scratch and save checkpoints.
    first_args = _clone_training_args(
        base_args,
        output_dir=health_output_dir,
        training_mode="steps",
        max_steps=first_run_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=2,
        save_checkpoints=True,
        resume_from_checkpoint=None,  # Explicitly no resume
        report_to="none",
        precision=chosen_precision,
    )

    first_trainer = KilatTrainer(
        model=model,
        args=first_args,
        train_dataset=health_train_dataset,
        eval_dataset=health_eval_dataset,
        data_collator=data_collator,
    )
    first_trainer.train()

    # Verify checkpoint was actually written.
    checkpoint_path = _find_numbered_checkpoint(health_output_dir)
    # HACK: Some trainer versions might not save optimizer state; we require it
    # because resume without it is unsafe for production.
    if not (checkpoint_path / "training_state.pt").exists():
        raise FileNotFoundError(
            f"Checkpoint missing training_state.pt: {checkpoint_path}"
        )

    # Fresh model to verify that all state (weights, optimizer, step) is restored.
    # WARNING: model.__class__ must support config‑only initialization.
    fresh_model = model.__class__(model.config)

    resume_args = _clone_training_args(
        base_args,
        output_dir=health_output_dir,
        training_mode="steps",
        max_steps=resumed_target_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        save_total_limit=2,
        save_checkpoints=True,
        resume_from_checkpoint=str(checkpoint_path),
        report_to="none",
        precision=chosen_precision,
    )

    resumed_trainer = KilatTrainer(
        model=fresh_model,
        args=resume_args,
        train_dataset=health_train_dataset,
        eval_dataset=health_eval_dataset,
        data_collator=data_collator,
    )
    resumed_trainer.train()

    # Success if both trainers achieved their target steps and the final
    # checkpoint (at resumed_target_steps) exists.
    success = (
        first_trainer.global_step >= first_run_steps
        and resumed_trainer.global_step >= resumed_target_steps
        and (Path(health_output_dir) / f"checkpoint-{resumed_target_steps}").exists()
    )

    return HealthCheckReport(
        success=success,
        output_dir=health_output_dir,
        first_run_steps=first_trainer.global_step,
        resumed_steps=resumed_trainer.global_step,
        checkpoint_path=str(checkpoint_path),
        notes=tuple(notes),
    )