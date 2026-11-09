"""Pillar 4: System Efficiency & Hardware Dynamics.

Evaluates sub-quadratic scaling advantages against baseline architectures:
1. Prefill Throughput (tokens/sec) across sequence lengths.
2. Step-by-step Decode Latency (ms/token and decode tokens/sec).
3. Peak VRAM Memory Footprint (MB) scaling.
4. Empirical Speedup vs Baseline (e.g. Vanilla Quadratic Attention).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..registry import extract_logits


@dataclass
class EfficiencyConfig:
    seq_lens: List[int] = field(default_factory=lambda: [128, 256, 512])
    batch_size: int = 4
    n_warmup: int = 3
    n_iters: int = 10
    n_decode_steps: int = 16
    vocab_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _sync(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def measure_prefill(model: nn.Module, cfg: EfficiencyConfig, seq_len: int) -> Dict[str, Any]:
    """Measure forward-pass prefill throughput."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), device=cfg.device)

    for _ in range(cfg.n_warmup):
        _ = model(x)
    _sync(cfg.device)

    t0 = time.perf_counter()
    for _ in range(cfg.n_iters):
        _ = model(x)
    _sync(cfg.device)
    elapsed = time.perf_counter() - t0

    tokens_per_iter = cfg.batch_size * seq_len
    total_tokens = tokens_per_iter * cfg.n_iters
    return {
        "seq_len": seq_len,
        "sec_per_prefill": elapsed / max(1, cfg.n_iters),
        "prefill_tokens_per_sec": total_tokens / max(1e-6, elapsed),
    }


@torch.no_grad()
def measure_decode(model: nn.Module, cfg: EfficiencyConfig, context_len: int) -> Dict[str, Any]:
    """Measure token-by-token generation latency."""
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, context_len), device=cfg.device)

    # Warmup
    _ = model(x)
    _sync(cfg.device)

    cur = x
    t0 = time.perf_counter()
    for _ in range(cfg.n_decode_steps):
        logits = extract_logits(model(cur))
        next_tok = logits[:, -1:, :].argmax(dim=-1)
        cur = torch.cat([cur, next_tok], dim=1)
    _sync(cfg.device)
    elapsed = time.perf_counter() - t0

    sec_per_token = elapsed / max(1, cfg.n_decode_steps)
    tokens_per_sec = (cfg.batch_size * cfg.n_decode_steps) / max(1e-6, elapsed)
    return {
        "context_len": context_len,
        "n_decode_steps": cfg.n_decode_steps,
        "sec_per_token": sec_per_token,
        "decode_ms_per_token": sec_per_token * 1000.0,
        "decode_tokens_per_sec": tokens_per_sec,
    }


def measure_peak_vram(model: nn.Module, cfg: EfficiencyConfig, seq_len: int) -> Dict[str, Any]:
    """Measure peak VRAM allocated during forward + backward passes."""
    if cfg.device != "cuda" or not torch.cuda.is_available():
        return {
            "seq_len": seq_len,
            "peak_vram_mb": None,
            "note": "CUDA not available (ran on CPU)",
        }

    torch.cuda.reset_peak_memory_stats(cfg.device)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, seq_len), device=cfg.device)
    y = torch.roll(x, shifts=-1, dims=1)

    logits = extract_logits(model(x))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    _sync(cfg.device)

    peak_mb = torch.cuda.max_memory_allocated(cfg.device) / (1024 ** 2)
    return {
        "seq_len": seq_len,
        "peak_vram_mb": float(peak_mb),
    }


def run_pillar4(
    model_builder: Callable[..., nn.Module],
    cfg: EfficiencyConfig,
    baseline_builder: Optional[Callable[..., nn.Module]] = None,
) -> Dict[str, Any]:
    """Execute Pillar 4 efficiency battery and compute speedup against baseline if provided."""
    # Ensure max_seq_len accommodates both prefill sweep and appended decode steps
    max_len = max(cfg.seq_lens) + cfg.n_decode_steps + 64
    model = model_builder(vocab_size=cfg.vocab_size, max_seq_len=max_len).to(cfg.device)
    sweep_entries = []

    for seq_len in cfg.seq_lens:
        entry = {
            "seq_len": seq_len,
            "prefill": measure_prefill(model, cfg, seq_len),
            "vram": measure_peak_vram(model, cfg, seq_len),
        }
        sweep_entries.append(entry)

    decode_entry = measure_decode(model, cfg, context_len=cfg.seq_lens[0])

    report: Dict[str, Any] = {
        "pillar": "P4_HARDWARE_EFFICIENCY",
        "passed": True,
        "seq_len_sweep": sweep_entries,
        "decode": decode_entry,
    }

    # Optional comparative benchmarking against baseline model
    if baseline_builder is not None:
        base_model = baseline_builder(vocab_size=cfg.vocab_size, max_seq_len=max_len).to(cfg.device)
        base_sweep = []
        for seq_len in cfg.seq_lens:
            base_sweep.append({
                "seq_len": seq_len,
                "prefill": measure_prefill(base_model, cfg, seq_len),
                "vram": measure_peak_vram(base_model, cfg, seq_len),
            })
        base_decode = measure_decode(base_model, cfg, context_len=cfg.seq_lens[0])

        comparison_table = []
        for m_entry, b_entry in zip(sweep_entries, base_sweep):
            m_tps = m_entry["prefill"]["prefill_tokens_per_sec"]
            b_tps = b_entry["prefill"]["prefill_tokens_per_sec"]
            speedup = m_tps / max(1e-6, b_tps)
            comparison_table.append({
                "seq_len": m_entry["seq_len"],
                "model_prefill_tps": m_tps,
                "baseline_prefill_tps": b_tps,
                "speedup": speedup,
            })

        report["baseline_comparison"] = {
            "sweep": comparison_table,
            "baseline_decode": base_decode,
        }

    return report
