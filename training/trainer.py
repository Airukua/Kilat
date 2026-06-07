from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Optional

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
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


def _iterable_from_dataset(ds: Dataset) -> IterableDataset:
    """Wrap a Dataset with an IterableDataset that delegates iteration.

    This forces DataLoader to use the dataset's __iter__ implementation
    (required for streaming mode where __getitem__ is not supported).
    
    Streaming datasets typically implement __iter__ but not __getitem__,
    making them incompatible with DataLoader's default map-style iteration.
    This wrapper bridges that gap by wrapping the dataset in an IterableDataset
    that simply delegates to the underlying iterator.
    """

    class _Wrapper(IterableDataset):
        def __init__(self, inner: Dataset):
            self._inner = inner

        def __iter__(self):
            return iter(self._inner)

    return _Wrapper(ds)


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
    tokenizer_model_path : Optional[str]
        Path to SentencePiece model file for decoding generated text samples
        during evaluation. If None, falls back to default path or raw token IDs.

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
        tokenizer_model_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self._prompt_decoder = self._load_prompt_decoder(tokenizer_model_path)

        # Reproducibility: Set seed before any initialization to ensure
        # consistent parameter initialization, data shuffling, and dropout patterns.
        # Using args.seed instead of a hardcoded value allows users to control
        # experiment reproducibility across different runs.
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
        # 'cuda' string is used for dispatch to the correct backend.
        # CPU AMP is not supported for training workflows.
        self._autocast_device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Training DataLoader: shuffle=True for training to prevent the model
        # from learning dataset order patterns. pin_memory speeds up CPU->GPU
        # transfers by using pinned (page-locked) memory.
        # Streaming datasets require special handling: they use __iter__ instead
        # of __getitem__, so we must either wrap them or configure DataLoader
        # with batch_size=None to accept pre-collated batches directly.
        is_streaming = getattr(self.train_dataset, 'streaming', False)
        self._train_is_streaming = is_streaming

        if is_streaming:
            # For streaming datasets, we wrap the dataset to force DataLoader
            # to use __iter__. batch_size=None tells DataLoader to yield elements
            # directly (the dataset already yields collated batches).
            ds_for_loader = _iterable_from_dataset(self.train_dataset)
            self.train_dataloader = DataLoader(
                ds_for_loader,
                batch_size=None,
                shuffle=False,  # Shuffling is handled by the streaming dataset itself
                collate_fn=None,  # Data is already collated by the streaming dataset
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

        # Evaluation DataLoader (optional) + early stopping callback
        # Early stopping uses a patience-based approach: training stops if
        # eval loss doesn't improve for 'patience' consecutive evaluations.
        # Threshold prevents stopping on negligible improvements (< threshold)
        # that could be attributed to noise rather than genuine overfitting.
        if self.eval_dataset is not None:
            is_streaming_eval = getattr(self.eval_dataset, "streaming", False)
            self._eval_is_streaming = is_streaming_eval
            if is_streaming_eval:
                eval_ds_for_loader = _iterable_from_dataset(self.eval_dataset)
                self.eval_dataloader: Optional[DataLoader] = DataLoader(
                    eval_ds_for_loader,
                    batch_size=None,
                    shuffle=False,  # Deterministic evaluation order
                    collate_fn=None,
                    pin_memory=torch.cuda.is_available(),
                )
            else:
                self.eval_dataloader: Optional[DataLoader] = DataLoader(
                    self.eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    shuffle=False,  # Deterministic results for reproducibility
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
            self._eval_is_streaming = False

        # Compute total steps for the scheduler.
        # In steps mode: directly uses max_steps (training budget is optimizer updates).
        # In epochs mode: calculates steps as num_epochs * batches_per_epoch / accumulation_steps
        #   because optimizer only steps after accumulation windows are complete.
        # For streaming datasets, we use dataset length if available; otherwise default to 1
        # (the actual iteration will be controlled by max_steps).
        if self._train_is_streaming:
            dataloader_len = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else 1
        else:
            dataloader_len = len(self.train_dataloader)

        self.total_steps = compute_total_steps(
            args.training_mode,
            args.max_steps,
            args.num_train_epochs,
            dataloader_len,
            args.gradient_accumulation_steps,
        )

        # Optimizer and scheduler setup.
        # AdamW is used for decoupled weight decay (Loshchilov & Hutter, 2019),
        # which separates weight decay from gradient-based updates, improving
        # generalization compared to L2 regularization in Adam.
        self.optimizer = create_optimizer(
            self.model,
            args.learning_rate,
            args.weight_decay,
        )
        
        # Cosine schedule with linear warmup: warmup prevents early training
        # instability when the model is far from optimum by gradually increasing LR.
        # Cosine decay then provides smooth LR reduction following the "SGDR" paper
        # approach (Loshchilov & Hutter, 2017), which has become standard for LLM training.
        self.scheduler = create_scheduler(
            self.optimizer,
            self.total_steps,
            args.warmup_steps,
        )

        # GradScaler for FP16 training stability.
        # The scaler multiplies loss by a dynamic scale factor before backward to prevent
        # gradient underflow in FP16, then unscales gradients before optimizer step.
        # 'cuda' device string is required even though the scaler is a no-op on CPU
        # (it's needed for API consistency with torch.amp).
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=self._scaler_enabled
        )

        # Global training state — these track progress and can be restored from checkpoint.
        # Initial values represent a fresh training start; resume_from_checkpoint
        # will override them if a checkpoint path is provided.
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.best_eval_loss: float = float("inf")  # Lower is better; tracks best model
        self.start_time: float = time.time()  # Used for throughput calculations

        # Resume from checkpoint if specified.
        # This enables preemption recovery: if a job is killed (common in SLURM/cluster
        # environments), the trainer can resume from the last checkpoint, restoring
        # model weights, optimizer state, scheduler state, AMP scaler state, and
        # early stopping counters — all necessary for exact training resumption.
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

        # Initialize WandB logging if configured.
        # We pass model_config to capture architecture details automatically
        # for experiment tracking and reproducibility across runs.
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

    def _load_prompt_decoder(self, tokenizer_model_path: Optional[str]) -> Optional[spm.SentencePieceProcessor]:
        """Load a SentencePiece model for decoding validation prompts into readable text.
        
        Tries the explicit path first, then falls back to a default location.
        Returns None if neither path exists, in which case raw token IDs are displayed.
        """
        if tokenizer_model_path is not None and os.path.exists(tokenizer_model_path):
            decoder = spm.SentencePieceProcessor()
            decoder.load(tokenizer_model_path)
            return decoder

        # Fallback to a conventional path within the project structure
        default_path = os.path.join(
            os.getcwd(),
            "data",
            "tokens",
            "train",
            "tokenizer",
            "sp_tokenizer.model",
        )
        if os.path.exists(default_path):
            decoder = spm.SentencePieceProcessor()
            decoder.load(default_path)
            return decoder

        return None

    def _decode_prompt(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to human-readable text for evaluation display.
        
        Falls back to displaying raw token IDs if no SentencePiece model is loaded,
        which still allows inspection but is less interpretable.
        """
        if self._prompt_decoder is None:
            text = " ".join(str(int(tok)) for tok in token_ids.tolist())
            return f"[token ids] {text}"

        return self._prompt_decoder.decode(token_ids.tolist())

    def _generate_sample(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50) -> str:
        """
        Generate text continuation from a prompt to show model capability during eval.
        
        Uses the model's generate method (standard HuggingFace interface) with
        nucleus sampling (top_p=0.9) and temperature=0.8 for diverse but coherent output.
        Falls back gracefully to an error message if generation fails.
        
        Args:
            prompt_ids: Token IDs tensor with shape (1, seq_len) — single prompt
            max_new_tokens: Maximum number of tokens to generate beyond the prompt
        
        Returns:
            Decoded text string containing the full generated sequence (prompt + continuation)
        """
        self.model.eval()
        
        with torch.inference_mode():
            try:
                # Use model's generate method — standard HuggingFace interface
                # Pad token ID is assumed to be 0; adjust if using a different tokenizer
                generated_ids = self.model.generate(
                    input_ids=prompt_ids.to(self.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=0,
                )
                
                generated_text = self._decode_prompt(generated_ids[0])
                return generated_text
            except Exception as e:
                # Graceful fallback if generate() is unavailable or fails
                return f"[generation failed: {str(e)}]"

    # -------------------------------------------------------------------
    # Main training loop — dispatcher
    # -------------------------------------------------------------------

    def train(self) -> None:
        """
        Run the training loop according to the selected ``training_mode``.
        
        Dispatches to either step-based or epoch-based training based on
        TrainingArguments.training_mode. Both paths share the same evaluation,
        checkpointing, and early stopping logic but differ in their outer loop
        structure and termination conditions.
        
        Handles KeyboardInterrupt gracefully by saving a checkpoint before exit,
        preserving all training state for later resumption.
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
        
        Designed for large-scale pretraining where:
        1. The dataset may be infinite (streaming) or too large for epoch counting
        2. Training duration is measured in optimizer updates, not data passes
        3. You want precise control over the total number of optimization steps
        
        The loop iterates through the dataloader indefinitely, cycling through
        epochs as needed, until the step budget is exhausted. Loss averaging is
        done over gradient accumulation windows (not epochs), giving a stable
        metric that reflects the effective batch size.
        """
        progress_bar = tqdm(
            total=self.total_steps,
            initial=self.global_step,  # Start from checkpoint position if resuming
            desc="Training (steps)",
            dynamic_ncols=True,
            unit="step",
        )

        step_within_accum: int = 0  # Counter for gradient accumulation window
        running_loss: float = 0.0  # Accumulated loss for current accumulation window

        # Start from the restored epoch (1 if fresh start, checkpoint value if resuming)
        epoch = self.current_epoch or 1
        while self.global_step < self.total_steps:
            self.current_epoch = epoch

            for batch in self.train_dataloader:
                loss_val = self._forward_backward(batch)
                running_loss += loss_val
                step_within_accum += 1

                # Only step the optimizer after accumulating enough gradients.
                # This simulates larger batch sizes without increasing memory:
                # e.g., batch_size=8 with accumulation_steps=4 gives effective batch of 32.
                if step_within_accum == self.args.gradient_accumulation_steps:
                    grad_norm = self._optimizer_step()
                    self.global_step += 1
                    step_within_accum = 0  # Reset for next accumulation window
                    current_lr = self.scheduler.get_last_lr()[0]

                    # Average loss over the accumulation window gives mean per-micro-batch loss
                    avg_loss = running_loss / self.args.gradient_accumulation_steps
                    
                    # PPL = exp(loss) is standard for language modeling.
                    # Capped at loss=100 to prevent overflow (exp(100) ≈ 2.7e43).
                    # In practice, loss > 10 indicates catastrophic training failure.
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

                    # Periodic logging: Log at configurable intervals to avoid
                    # overwhelming the logging backend while capturing training trajectory.
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

                    # Periodic evaluation: Only runs when eval dataset exists and at
                    # specified intervals. Evaluation is expensive (full pass through
                    # eval set), so we don't do it too frequently.
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        should_stop = self._run_eval_and_check_stopping()
                        if should_stop:
                            progress_bar.close()
                            return

                    # Periodic checkpointing for fault tolerance and model selection
                    if self.args.save_checkpoints and self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(self.global_step)

                    # Check termination: placed after optimizer step (not before)
                    # to ensure we complete the current optimization before stopping.
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
        
        Designed for fine-tuning and smaller datasets where:
        1. The dataset size is known and fixed
        2. You want to control the number of full passes through the data
        3. Each epoch represents a complete pass through the training set
        
        Key difference from steps mode: The outer loop is epoch-based for clarity,
        though total_steps is still computed for the scheduler. Evaluation and
        checkpointing occur at step intervals within epochs (not just epoch boundaries)
        to provide timely feedback on large datasets.
        """
        start_epoch = self.current_epoch or 1

        for epoch in range(start_epoch, self.args.num_train_epochs + 1):
            self.current_epoch = epoch

            # Each epoch gets its own progress bar for cleaner visualization.
            # 'leave=True' keeps completed epoch bars visible for reference.
            # For streaming datasets, we try to get dataset length; otherwise
            # the progress bar shows no total (unknown length).
            if self._train_is_streaming:
                total_batches = len(self.train_dataset) if hasattr(self.train_dataset, '__len__') else None
            else:
                total_batches = len(self.train_dataloader)

            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=total_batches,
                desc=f"Epoch {epoch}/{self.args.num_train_epochs}",
                dynamic_ncols=True,
                unit="batch",
                leave=True,
            )

            epoch_loss: float = 0.0  # Running loss for logging intervals within epoch
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
                    
                    # Average loss over all batches processed so far in this epoch
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

                    # Periodic logging within epoch (same logic as steps mode)
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


    def _forward_backward(self, batch: Any) -> float:
        """
        Run a single forward + backward pass with AMP autocast.
        
        Returns the loss value BEFORE scaling and accumulation division,
        so callers can accumulate it properly. The returned loss is
        multiplied by gradient_accumulation_steps to recover the original
        (pre-normalization) per-batch loss.
        
        Key design decisions:
        - Loss is divided by gradient_accumulation_steps BEFORE backward.
          This ensures the accumulated gradient is the mean of micro-batch
          gradients, not the sum, which is equivalent to training with a
          proportionally larger batch size.
        - non_blocking=True for device transfers overlaps data movement
          with computation, hiding CPU->GPU transfer latency.
        - autocast handles mixed precision conversion automatically based
          on the configured dtype and device type.
        
        Supports both dict-style batches (from DataLoader with collator)
        and tuple-style batches (from streaming IterableDatasets).
        """
        # Support both mapping batches (dict with 'input_ids'/'labels')
        # and streaming IterableDatasets that return (inputs, labels) tuples.
        if isinstance(batch, (tuple, list)):
            input_ids, labels = batch
            # DataLoader may add a leading batch dimension when using
            # batch_size=1 for streaming. Squeeze that dim if present.
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 3:
                input_ids = input_ids.squeeze(0)
                labels = labels.squeeze(0)
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # AMP autocast context: Casts operations to the specified precision
        # where beneficial (e.g., matmul in FP16/BF16) while keeping sensitive
        # operations (e.g., softmax, layernorm) in FP32 for numerical stability.
        # This follows the "Mixed Precision Training" paper (Micikevicius et al., 2018).
        with torch.amp.autocast(
            device_type=self._autocast_device,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
            # Normalize loss by gradient accumulation steps to get mean gradient
            loss = outputs.loss / self.args.gradient_accumulation_steps

        # Scale the loss before backward for FP16 training stability.
        # In BF16/FP32 modes, scaler.scale is a no-op.
        self.scaler.scale(loss).backward()

        # Return the original (unscaled) loss for logging/metrics.
        # Multiplying by accumulation steps recovers the true per-batch loss.
        return loss.item() * self.args.gradient_accumulation_steps

    def _optimizer_step(self) -> torch.Tensor:
        """
        Execute a single optimizer step: unscale gradients, clip, update weights.
        
        Returns the gradient norm BEFORE clipping for monitoring purposes.
        This helps detect gradient explosion issues during training.
        
        The step sequence follows the prescribed order for AMP training:
        1. unscale_: Reverses loss scaling to recover true gradients
        2. clip_grad_norm_: Prevents gradient explosion (common in transformers)
        3. scaler.step: Updates weights (may skip if gradients contain infs/nans)
        4. scaler.update: Adjusts loss scale for next iteration
        5. scheduler.step: Updates learning rate
        6. zero_grad(set_to_none=True): Releases gradient memory entirely
           rather than filling with zeros, which is more memory-efficient
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
        - Updates self.best_eval_loss if current eval loss improves
        - Saves best checkpoint when a new best is found
        - Logs early stopping message and saves final checkpoint if triggered
        - CRITICAL: Restores model to training mode after evaluation
          (evaluate() sets model.eval(), which disables dropout/batch norm)
        """
        eval_loss, eval_ppl = self.evaluate()
        
        # Restore training mode — evaluate() disables dropout/batch norm.
        # Forgetting this would silently degrade training quality.
        self.model.train()

        # Track best model: Only save checkpoint when we find a new best
        # to prevent checkpoint bloat while preserving the best-performing state.
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            if self.args.save_checkpoints:
                self._save_checkpoint(self.global_step, tag="best")

        # Early stopping: Uses patience-based approach to prevent stopping
        # on temporary loss spikes while still catching genuine overfitting.
        # The callback tracks consecutive evaluations without improvement
        # and signals stop when patience is exhausted.
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
        
        Uses inference_mode() instead of no_grad() because inference_mode()
        provides additional optimizations by disabling autograd version tracking
        entirely (~5-10% speedup vs no_grad).
        
        Also selects a random prompt from the eval set and generates a sample
        continuation to provide qualitative assessment of model capability.
        
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

        # Reservoir sampling: uniformly select one random prompt from the eval set
        # to display for qualitative inspection. This gives a representative sample
        # without biasing toward early or late batches.
        selected_prompt: Optional[torch.Tensor] = None
        selected_count = 0

        for batch in eval_progress:
            # Support both dict batches and (inputs, labels) tuples from streaming datasets
            if isinstance(batch, (tuple, list)):
                input_ids, labels = batch
                if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 3:
                    input_ids = input_ids.squeeze(0)
                    labels = labels.squeeze(0)
            else:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

            # Reservoir sampling: each sequence has 1/count chance of being selected
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            for seq in input_ids.detach().cpu():
                selected_count += 1
                if random.randrange(selected_count) == 0:
                    selected_prompt = seq

            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self._autocast_device,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(input_ids=input_ids, labels=labels, return_dict=True)
                eval_loss += outputs.loss.item()

            # Update progress bar with current loss for real-time monitoring
            eval_progress.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        # Compute average loss across all batches.
        # max(1, ...) prevents division by zero for empty dataloader edge case.
        if self._eval_is_streaming:
            num_eval_batches = len(self.eval_dataset) if self.eval_dataset is not None and hasattr(self.eval_dataset, '__len__') else 1
        else:
            num_eval_batches = len(self.eval_dataloader)

        avg_eval_loss = eval_loss / max(1, num_eval_batches)
        eval_ppl = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float("inf")

        # # Display sampled prompt and generated continuation for qualitative assessment
        # if selected_prompt is not None:
        #     prompt_text = self._decode_prompt(selected_prompt)
        #     print(f"\n[Eval sample prompt] {prompt_text}")
            
        #     generated_text = self._generate_sample(selected_prompt.unsqueeze(0))
        #     print(f"[Eval sample generated] {generated_text}")

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
        Save training checkpoint with automatic pruning of old checkpoints.
        
        The pruning mechanism (save_total_limit) prevents unbounded disk usage
        by keeping only the N most recent checkpoints. This is critical for
        long-running training where each checkpoint can be gigabytes.
        
        Checkpoint contents include:
        - Model weights (via save_pretrained for HuggingFace compatibility)
        - Optimizer state (for exact training resumption)
        - Scheduler state (to continue LR schedule from same point)
        - AMP scaler state (for FP16 stability continuity across restarts)
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
        Final cleanup: log summary metrics and close WandB connection.
        
        Called in all exit paths (normal completion, early stopping,
        interruption) to ensure consistent logging and resource cleanup
        regardless of how training terminates.
        """
        log_final_summary(
            self.global_step,
            self.start_time,
            self.best_eval_loss,
            self.args.output_dir,
        )
        finish_wandb(self.args.report_to)