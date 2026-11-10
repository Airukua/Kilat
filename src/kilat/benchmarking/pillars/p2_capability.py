"""Pillar 2: Mechanistic Retrieval & Associative Capacity.

Evaluates sub-quadratic model architectures on synthetic reasoning benchmarks:
1. Multi-Query Associative Recall (MQAR, Zoology ICLR 2024).
2. Leak-Free Induction Heads (Olsson et al., 2022; Mamba Appendix E.1).
3. Delimited Selective Copying (Mamba Appendix E.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..protocol import run_with_seeds, summarize
from ..registry import extract_logits
from ..tasks import induction_heads, mqar, selective_copying


@dataclass
class CapabilityConfig:
    task: str = "mqar"  # "mqar" | "induction_heads" | "selective_copying"
    context_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    train_steps: int = 400
    lr: float = 2e-3
    vocab_size: int = 128
    batch_size: int = 32
    d_model: Optional[int] = None
    device: str = "cpu"
    pass_threshold: float = 0.80


def _train_and_eval_single(
    model_builder: Callable[..., nn.Module],
    cfg: CapabilityConfig,
    seq_len: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_len = max(cfg.context_lengths)
    builder_kwargs: Dict[str, Any] = {
        "vocab_size": cfg.vocab_size,
        "max_seq_len": max_len,
    }
    if cfg.d_model is not None:
        builder_kwargs["d_model"] = cfg.d_model

    model = model_builder(**builder_kwargs).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.train_steps))

    # Resolve task generator and loss functions
    if cfg.task == "mqar":
        num_kv = max(4, seq_len // 16)
        task_cfg = mqar.MQARConfig(
            seq_len=seq_len,
            vocab_size=cfg.vocab_size,
            num_kv_pairs=num_kv,
            batch_size=cfg.batch_size,
        )

        n_train_examples = max(cfg.batch_size * 4, min(cfg.train_steps * cfg.batch_size, 3000))
        n_eval_examples = max(cfg.batch_size, 300)

        x_train, y_train = mqar.generate_dataset(
            task_cfg, num_examples=n_train_examples, seed=seed, device=cfg.device
        )
        x_eval, y_eval = mqar.generate_dataset(
            task_cfg, num_examples=n_eval_examples, seed=10_000 + seed, device=cfg.device
        )

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        loader_iter = iter(train_loader)

        loss_fn = mqar.compute_causal_loss
        acc_fn = mqar.compute_causal_accuracy

        model.train()
        for _ in range(cfg.train_steps):
            try:
                bx, by = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                bx, by = next(loader_iter)

            opt.zero_grad()
            logits = extract_logits(model(bx))
            loss = loss_fn(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            eval_logits = extract_logits(model(x_eval))
            final_acc = acc_fn(eval_logits, y_eval)
            final_loss = loss_fn(eval_logits, y_eval).item()

        return {
            "accuracy": final_acc,
            "loss": final_loss,
        }

    elif cfg.task == "induction_heads":
        task_cfg = induction_heads.InductionHeadsConfig(
            vocab_size=cfg.vocab_size,
            seq_len=seq_len,
            batch_size=cfg.batch_size,
        )
        gen_train = torch.Generator().manual_seed(seed)
        gen_eval = torch.Generator().manual_seed(10_000 + seed)

        def gen_fn(g):
            return induction_heads.generate_batch(task_cfg, device=cfg.device, generator=g)

        loss_fn = induction_heads.compute_causal_loss
        acc_fn = induction_heads.compute_causal_accuracy

    elif cfg.task == "selective_copying":
        task_cfg = selective_copying.SelectiveCopyingConfig(
            seq_len=seq_len,
            vocab_size=cfg.vocab_size,
            num_data_tokens=min(8, cfg.vocab_size - 4),
            batch_size=cfg.batch_size,
        )
        gen_train = torch.Generator().manual_seed(seed)
        gen_eval = torch.Generator().manual_seed(10_000 + seed)

        def gen_fn(g):
            return selective_copying.generate_batch(task_cfg, device=cfg.device, generator=g)

        loss_fn = selective_copying.compute_causal_loss
        acc_fn = selective_copying.compute_causal_accuracy
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    # Training loop for streaming tasks (induction heads, selective copying)
    model.train()
    for _ in range(cfg.train_steps):
        x, y = gen_fn(gen_train)
        opt.zero_grad()
        logits = extract_logits(model(x))
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

    # Evaluation on held-out distribution
    model.eval()
    with torch.no_grad():
        x_eval, y_eval = gen_fn(gen_eval)
        eval_logits = extract_logits(model(x_eval))
        final_acc = acc_fn(eval_logits, y_eval)
        final_loss = loss_fn(eval_logits, y_eval).item()

    return {
        "accuracy": final_acc,
        "loss": final_loss,
    }


def run_pillar2(
    model_builder: Callable[..., nn.Module],
    cfg: CapabilityConfig,
    arch_name: str = "model",
) -> Dict[str, Any]:
    """Execute Pillar 2 sweep across context lengths and seeds."""
    grid_results: Dict[int, Dict[str, Any]] = {}

    for seq_len in cfg.context_lengths:
        seed_results = run_with_seeds(
            lambda s, sl=seq_len: _train_and_eval_single(model_builder, cfg, sl, s),
            seeds=cfg.seeds,
        )
        grid_results[seq_len] = summarize(seed_results)

    # Overall pass criterion: average accuracy at shortest context >= pass_threshold
    base_acc = grid_results[cfg.context_lengths[0]].get("accuracy", {}).get("mean", 0.0)
    passed = base_acc >= cfg.pass_threshold

    return {
        "pillar": "P2_MECHANISTIC_CAPABILITY",
        "task": cfg.task,
        "arch_name": arch_name,
        "passed": passed,
        "grid": {str(k): v for k, v in grid_results.items()},
        "context_lengths": cfg.context_lengths,
        "train_steps": cfg.train_steps,
    }
