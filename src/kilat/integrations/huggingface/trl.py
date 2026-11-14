"""
Kilat Post-Training Alignment Bridge (TRL SFT & DPO Compatibility — Phase P2).

Provides thin, zero-duplication utilities for Supervised Fine-Tuning (SFT) and
Direct Preference Optimization (DPO) on Kilat models:
1. Cache deactivation (`use_cache = False`) and non-reentrant activation checkpointing
   (`gradient_checkpointing_kwargs={"use_reentrant": False}`) with input-embedding
   gradient hooks for PEFT LoRA compatibility.
2. Completion-only prompt loss masking (`KilatCompletionOnlyCollator`) enforcing
   strict `-100` masking on user prompts and padding tokens (Golden Invariant #3).
3. Numerically stable FP32 causal sequence log-probability and DPO preference loss
   primitives (`compute_sequence_logprobs`, `compute_dpo_preference_loss`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kilat.assurance.types import unwrap_kilat_model

IGNORE_INDEX = -100


def prepare_kilat_for_post_training(
    model: nn.Module,
    use_gradient_checkpointing: bool = True,
    use_reentrant: bool = False,
) -> nn.Module:
    """
    Prepare a Kilat model (or PEFT-wrapped Kilat model) for SFT / DPO alignment.

    Enforces:
      - `model.config.use_cache = False` on both wrapper and base model.
      - Input embedding gradient hook (`enable_input_require_grads()`) when frozen
        base embeddings are combined with gradient checkpointing.
      - Non-reentrant activation checkpointing (`use_reentrant=False`).
    """
    base_model = unwrap_kilat_model(model)
    for target in (model, base_model):
        cfg = getattr(target, "config", None)
        if cfg is not None:
            setattr(cfg, "use_cache", False)

    if use_gradient_checkpointing:
        for target in (model, base_model):
            enable_grads_fn = getattr(target, "enable_input_require_grads", None)
            if callable(enable_grads_fn):
                enable_grads_fn()
        ckpt_kwargs = {"use_reentrant": bool(use_reentrant)}
        ckpt_enable_fn = getattr(
            base_model,
            "gradient_checkpointing_enable",
            getattr(model, "gradient_checkpointing_enable", None),
        )
        if callable(ckpt_enable_fn):
            ckpt_enable_fn(gradient_checkpointing_kwargs=ckpt_kwargs)
    return model


class KilatCompletionOnlyCollator:
    """
    Data collator for Supervised Fine-Tuning (SFT) that masks all prompt tokens
    up to and including `response_template_ids` (as well as padding tokens) with `-100`.

    Compatible with Hugging Face `Trainer`, `trl.SFTTrainer`, and `KilatTrainer`.
    """

    def __init__(
        self,
        response_template_ids: Sequence[int],
        pad_token_id: int = 0,
        max_length: Optional[int] = None,
    ) -> None:
        if not response_template_ids:
            raise ValueError("response_template_ids must be a non-empty sequence of token IDs.")
        self.response_template_ids = [int(t) for t in response_template_ids]
        self.pad_token_id = int(pad_token_id)
        self.max_length = max_length

    def _find_response_start(self, token_ids: List[int]) -> Optional[int]:
        tpl = self.response_template_ids
        tpl_len = len(tpl)
        for idx in range(len(token_ids) - tpl_len + 1):
            if token_ids[idx : idx + tpl_len] == tpl:
                return idx + tpl_len
        return None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("KilatCompletionOnlyCollator received an empty batch.")

        sequences: List[List[int]] = []
        for feat in features:
            raw_ids = feat["input_ids"]
            seq = raw_ids.tolist() if isinstance(raw_ids, torch.Tensor) else [int(x) for x in raw_ids]
            if self.max_length is not None:
                seq = seq[: self.max_length]
            sequences.append(seq)

        max_seq_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)

        for row_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            input_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row_idx, :seq_len] = 1

            resp_start = self._find_response_start(seq)
            if resp_start is None or resp_start >= seq_len:
                raise ValueError(
                    f"Sample #{row_idx} does not contain response_template_ids {self.response_template_ids} "
                    "followed by at least one completion token."
                )
            labels[row_idx, resp_start:seq_len] = input_ids[row_idx, resp_start:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_sequence_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Compute causal per-sequence log-probabilities over non-masked target tokens (`labels != -100`).

    Uses FP32 log-softmax for numerical stability across FP16/BF16 mixed-precision runs.
    """
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:].clone()

    loss_mask = shift_labels != IGNORE_INDEX
    safe_labels = shift_labels.clamp(min=0)

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    masked_logps = per_token_logps * loss_mask.to(per_token_logps.dtype)

    seq_logps = masked_logps.sum(dim=-1)
    if average_log_prob:
        denom = loss_mask.sum(dim=-1).clamp(min=1).to(seq_logps.dtype)
        return seq_logps / denom
    return seq_logps


def compute_dpo_preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Direct Preference Optimization (DPO) loss and implicit rewards (Rafailov et al., 2023).

    Returns `(loss, chosen_rewards, rejected_rewards)` where:
        chosen_rewards   = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    """
    if beta <= 0.0:
        raise ValueError(f"DPO temperature beta must be > 0, got {beta}.")
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError(f"label_smoothing must be in [0.0, 0.5), got {label_smoothing}.")

    chosen_logratios = policy_chosen_logps.float() - ref_chosen_logps.float().detach()
    rejected_logratios = policy_rejected_logps.float() - ref_rejected_logps.float().detach()
    logits = beta * (chosen_logratios - rejected_logratios)

    losses = (
        -F.logsigmoid(logits) * (1.0 - label_smoothing)
        - F.logsigmoid(-logits) * label_smoothing
    )
    chosen_rewards = (beta * chosen_logratios).detach()
    rejected_rewards = (beta * rejected_logratios).detach()
    return losses.mean(), chosen_rewards, rejected_rewards
