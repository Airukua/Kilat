"""
Supervised Fine-Tuning (SFT) Trainer for Kilat.

Extends KilatTrainer with automatic SFT dataset collation, instruction loss masking,
and fine-tuning optimization heuristics.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kilat.data.sft import SFTDataCollator, SFTDataset
from kilat.engine.args import TrainingArguments
from kilat.engine.trainer import KilatTrainer
from kilat.utils.banner import print_kilat_banner


class KilatSFTTrainer(KilatTrainer):
    """
    Specialized Trainer for Supervised Fine-Tuning (SFT / Instruction Tuning).

    Simplifies fine-tuning workflows by:
    - Automatically wrapping datasets with ``SFTDataCollator`` (selective prompt loss masking)
    - Handling raw JSONL, list of dictionaries, or PyTorch Datasets directly
    - Preserving all Kilat pre-training guarantees (atomic checkpointing, mixed precision, MoE stability)

    Parameters
    ----------
    model : nn.Module
        The pre-trained KilatTransformer to fine-tune.
    args : TrainingArguments
        Hyperparameters and training configuration.
    train_dataset : Optional[Union[Dataset, List[Dict[str, Any]], str]]
        Instruction dataset (Alpaca, ChatML, prompt-response).
    eval_dataset : Optional[Union[Dataset, List[Dict[str, Any]], str]]
        Validation dataset.
    tokenizer : Any
        Tokenizer instance matching the model vocabulary.
    data_collator : Optional[Any]
        Custom collator. If None, defaults to ``SFTDataCollator(tokenizer, max_length=max_seq_length)``.
    max_seq_length : int
        Maximum context length for collation (default: 512).
    callbacks : Optional[list]
        Additional TrainerCallback instances.
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Union[Dataset, List[Dict[str, Any]], str]] = None,
        eval_dataset: Optional[Union[Dataset, List[Dict[str, Any]], str]] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        max_seq_length: int = 512,
        callbacks: Optional[list] = None,
        eval_fn: Optional[Callable] = None,
        compute_loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        if tokenizer is None:
            raise ValueError("A valid tokenizer is required for KilatSFTTrainer.")

        # Auto-place model on hardware accelerator if available and currently on CPU
        if torch.cuda.is_available():
            current_device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            if current_device.type != "cuda":
                target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                model = model.to(target_device)

        # 1. Resolve Data Collator
        if data_collator is None:
            data_collator = SFTDataCollator(
                tokenizer=tokenizer,
                max_length=max_seq_length,
                ignore_index=-100,
                append_eos_token=True,
            )
        self.data_collator = data_collator

        # 2. Resolve Train DataLoader
        if train_dataloader is None:
            if train_dataset is None:
                raise ValueError("Must provide either 'train_dataset' or 'train_dataloader'.")
            if not isinstance(train_dataset, Dataset):
                train_dataset = SFTDataset(train_dataset)

            batch_size = int(getattr(args, "per_device_train_batch_size", 8))
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )

        # 3. Resolve Eval DataLoader
        if eval_dataloader is None and eval_dataset is not None:
            if not isinstance(eval_dataset, Dataset):
                eval_dataset = SFTDataset(eval_dataset)

            eval_batch_size = int(getattr(args, "per_device_eval_batch_size", 8))
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=False,
                pin_memory=torch.cuda.is_available(),
            )

        super().__init__(
            model=model,
            args=args,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            eval_fn=eval_fn,
            compute_loss_fn=compute_loss_fn,
            callbacks=callbacks,
            tokenizer=tokenizer,
            **kwargs,
        )

        self._log_sft_summary(train_dataloader, max_seq_length)

    def _log_sft_summary(self, train_loader: DataLoader, max_seq_len: int) -> None:
        """Log SFT training configuration summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        eff_batch = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps

        sys.stdout.write(
            f"\n⚡ [KILAT SFT INITIALIZED]\n"
            f"  - Model Parameters:     {total_params:,} (Trainable: {trainable_params:,})\n"
            f"  - Max Sequence Length:  {max_seq_len}\n"
            f"  - Micro Batch Size:     {self.args.per_device_train_batch_size}\n"
            f"  - Effective Batch Size: {eff_batch}\n"
            f"  - Learning Rate:        {self.args.learning_rate:.2e}\n"
            f"  - Mixed Precision:      {self.args.precision}\n\n"
        )
        sys.stdout.flush()
