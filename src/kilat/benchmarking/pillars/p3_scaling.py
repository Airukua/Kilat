"""Pillar 3: Empirical Scaling Laws & Quality.

Evaluates scaling behavior on token-budget controlled language modeling:
1. Strict Token-Budget Control: Compares models on identical token budgets, not fixed steps.
2. Validation Perplexity (PPL) Scaling Curve.
3. Power-Law Exponent Estimation: Fits L(N) ~ A * N^(-alpha) to check power-law compliance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import extract_logits


@dataclass
class ScalingConfig:
    seq_lens: List[int] = field(default_factory=lambda: [128, 256])
    token_budgets: List[int] = field(default_factory=lambda: [50_000])
    eval_every_tokens: int = 10_000
    lr: float = 5e-4
    # Default synthetic data: Markov chain where context matters (see markov_entropies).
    markov_branching: int = 4
    markov_chain_seed: int = 1234
    # Pass iff the model closes at least this fraction of the gap between the context-free floor
    # (unigram entropy) and the Bayes-optimal floor (true conditional entropy).
    min_gap_closed: float = 0.5
    vocab_size: int = 256
    batch_size: int = 16
    device: str = "cpu"


def _markov_transition(vocab_size: int, branching: int, chain_seed: int) -> torch.Tensor:
    """Sparse first-order Markov chain: each token has `branching` successors with Dirichlet(1) weights.

    Unlike i.i.d. tokens, next-token prediction here *requires* context, so architectures can be separated.
    """
    gen = torch.Generator().manual_seed(chain_seed)
    trans = torch.zeros(vocab_size, vocab_size)
    k = max(1, min(branching, vocab_size))
    for i in range(vocab_size):
        succ = torch.randperm(vocab_size, generator=gen)[:k]
        w = -torch.log(torch.rand(k, generator=gen).clamp_min(1e-12))  # Exp(1) => normalized = Dirichlet(1)
        trans[i, succ] = w / w.sum()
    return trans


def _stationary(trans: torch.Tensor, iters: int = 500) -> torch.Tensor:
    pi = torch.full((trans.size(0),), 1.0 / trans.size(0), dtype=torch.float64)
    t64 = trans.to(torch.float64)
    for _ in range(iters):
        pi = pi @ t64
    return pi / pi.sum()


def markov_entropies(vocab_size: int, branching: int = 4, chain_seed: int = 1234) -> Tuple[float, float]:
    """Return (unigram entropy, true conditional entropy) in nats for the default synthetic chain.

    unigram entropy  = best loss achievable WITHOUT using context
    conditional H    = Bayes-optimal loss WITH context (irreducible floor)
    """
    trans = _markov_transition(vocab_size, branching, chain_seed).to(torch.float64)
    pi = _stationary(trans)
    h_uni = float(-(pi[pi > 0] * pi[pi > 0].log()).sum())
    row_h = -(torch.where(trans > 0, trans * trans.clamp_min(1e-300).log(), torch.zeros_like(trans))).sum(dim=1)
    h_cond = float((pi * row_h).sum())
    return h_uni, h_cond


def _default_synthetic_stream(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    seed: int = 42,
    chain_seed: int = 1234,
    branching: int = 4,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Infinite stream of TWO independent realisations of a fixed Markov chain, interleaved: a1 b1 a2 b2 ...

    The next token depends on the token two positions back and is independent of the current token, so:
    - a model that only sees the current token (no sequence mixing) is floored at the unigram entropy;
    - only genuine context use reaches the conditional-entropy floor.
    `chain_seed` fixes the chain (shared by train/val); `seed` drives sampling. y[:, t] = x_{t+1}, no wrap-around.
    """
    trans = _markov_transition(vocab_size, branching, chain_seed)
    pi = _stationary(trans).to(torch.float32)
    gen = torch.Generator().manual_seed(seed)
    half = (seq_len + 1) // 2 + 1
    while True:
        chains = torch.empty(2, batch_size, half, dtype=torch.long)
        for c in range(2):
            chains[c, :, 0] = torch.multinomial(pi, batch_size, replacement=True, generator=gen)
            for t in range(1, half):
                chains[c, :, t] = torch.multinomial(trans[chains[c, :, t - 1]], 1, generator=gen).squeeze(1)
        seq = torch.stack([chains[0], chains[1]], dim=-1).reshape(batch_size, 2 * half)[:, : seq_len + 1]
        yield seq[:, :-1].to(device), seq[:, 1:].to(device)


def train_with_token_budget(
    model: nn.Module,
    data_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: ScalingConfig,
    token_budget: int,
) -> Dict[str, Any]:
    """Train model until exactly token_budget tokens have been processed."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    tokens_seen = 0
    next_eval_at = cfg.eval_every_tokens
    eval_points: List[Dict[str, Any]] = []
    ema_loss: Optional[float] = None
    t0 = time.perf_counter()

    model.train()
    for x, y in data_iter:
        if tokens_seen >= token_budget:
            break
        opt.zero_grad()
        logits = extract_logits(model(x))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        batch_tokens = x.numel()
        tokens_seen += batch_tokens
        current_loss = loss.item()
        ema_loss = current_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * current_loss

        if tokens_seen >= next_eval_at:
            eval_points.append({
                "tokens_seen": tokens_seen,
                "train_loss_ema": ema_loss,
            })
            next_eval_at += cfg.eval_every_tokens

    wall_time = time.perf_counter() - t0
    return {
        "tokens_seen": tokens_seen,
        "wall_time_sec": wall_time,
        "tokens_per_sec": tokens_seen / max(1e-6, wall_time),
        "final_loss_ema": ema_loss,
        "eval_points": eval_points,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    data_iter: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: ScalingConfig,
    n_batches: int = 10,
) -> Dict[str, Any]:
    """Evaluate NLL and perplexity on held-out tokens."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(data_iter):
        if i >= n_batches:
            break
        logits = extract_logits(model(x))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid = (y != -100).sum().item()
        total_loss += loss.item()
        total_tokens += valid

    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_nll, 20.0))
    return {
        "nll": avg_nll,
        "perplexity": ppl,
        "tokens_evaluated": total_tokens,
    }


def compute_power_law_alpha(eval_points: List[Dict[str, Any]]) -> float:
    """Fit log(loss) = -alpha * log(tokens) + c using linear regression."""
    if len(eval_points) < 2:
        return 0.0
    tokens = np.array([pt["tokens_seen"] for pt in eval_points], dtype=np.float64)
    losses = np.array([pt["train_loss_ema"] for pt in eval_points], dtype=np.float64)
    log_t = np.log(tokens)
    log_l = np.log(np.maximum(losses, 1e-6))
    # slope = cov(log_t, log_l) / var(log_t)
    poly = np.polyfit(log_t, log_l, 1)
    slope = poly[0]
    return float(-slope)  # alpha is positive when loss decreases with tokens


def run_pillar3(
    model_builder: Callable[..., nn.Module],
    cfg: ScalingConfig,
    train_data_fn: Optional[Callable[[int], Iterable[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    val_data_fn: Optional[Callable[[int], Iterable[Tuple[torch.Tensor, torch.Tensor]]]] = None,
) -> Dict[str, Any]:
    """Execute Pillar 3 scaling evaluation.

    Pass criterion (never vacuous):
    - default Markov data: eval NLL must close >= `min_gap_closed` of the gap between unigram entropy
      (no context) and the true conditional entropy (perfect context use);
    - custom data: eval NLL must improve on the untrained model's NLL by >= 10%.
    """
    runs = []
    max_len = max(cfg.seq_lens)
    use_default = train_data_fn is None
    h_uni, h_cond = markov_entropies(cfg.vocab_size, cfg.markov_branching, cfg.markov_chain_seed)

    def _stream(seq_len: int, seed: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        return _default_synthetic_stream(
            cfg.vocab_size, seq_len, cfg.batch_size, cfg.device, seed=seed,
            chain_seed=cfg.markov_chain_seed, branching=cfg.markov_branching,
        )

    for seq_len in cfg.seq_lens:
        for budget in cfg.token_budgets:
            model = model_builder(vocab_size=cfg.vocab_size, max_seq_len=max_len).to(cfg.device)

            def _val() -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
                return val_data_fn(seq_len) if val_data_fn else _stream(seq_len, seed=999)

            init_eval = evaluate_perplexity(model, _val(), cfg)
            train_iter = train_data_fn(seq_len) if train_data_fn else _stream(seq_len, seed=123)
            train_report = train_with_token_budget(model, train_iter, cfg, budget)
            eval_report = evaluate_perplexity(model, _val(), cfg)
            alpha = compute_power_law_alpha(train_report["eval_points"])

            nll = float(eval_report["nll"])
            if use_default and val_data_fn is None:
                gap = max(1e-9, h_uni - h_cond)
                gap_closed = (h_uni - nll) / gap
                run_passed = math.isfinite(nll) and gap_closed >= cfg.min_gap_closed
            else:
                gap_closed = float("nan")
                run_passed = math.isfinite(nll) and nll <= 0.9 * float(init_eval["nll"])

            runs.append({
                "seq_len": seq_len,
                "token_budget": budget,
                "train": train_report,
                "init_eval": init_eval,
                "eval": eval_report,
                "scaling_alpha": alpha,
                "gap_closed": gap_closed,
                "passed": run_passed,
            })

    return {
        "pillar": "P3_EMPIRICAL_SCALING_LAWS",
        "passed": bool(runs) and all(r["passed"] for r in runs),
        "data": {
            "source": "markov" if use_default else "custom",
            "unigram_entropy_nats": h_uni,
            "conditional_entropy_nats": h_cond,
            "min_gap_closed": cfg.min_gap_closed,
        },
        "runs": runs,
    }
