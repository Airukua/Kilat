#!/usr/bin/env python3
"""Fair context-scaling benchmark: Kilat (dense FFN) vs LLaMA (dense) at parameter parity.

Fairness contract (printed and saved to JSON):
  * Same depth / hidden size / heads / vocab / tied embeddings / SwiGLU FFN family.
  * LLaMA ``intermediate_size`` is solved so its parameter count is the largest value that does
    NOT exceed Kilat's (Kilat is equal or slightly *heavier*, which is conservative for Kilat).
  * Same dtype, batch, seed, warm-up/iteration counts, CUDA sync, ``torch.inference_mode``.
  * LLaMA uses ``sdpa`` (fused/flash kernels) so the baseline is the strongest standard one.
  * OOM is recorded as a result (not a crash), so sweeps can go as long as the GPU allows.

Usage (Colab T4):
    python tools/fair_scaling_benchmark.py --lengths 512 1024 2048 4096 8192 16384 32768
    python tools/fair_scaling_benchmark.py --train --lengths 512 1024 2048 4096
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kilat.configs.model_config import KilatConfig
from kilat.models.transformer import KilatTransformer

TOLERANCE_PCT = 0.1  # max allowed |param diff| in percent (granularity is ~0.03%)


def build_kilat(args: argparse.Namespace, max_len: int) -> KilatTransformer:
    cfg = KilatConfig(
        vocab_size=args.vocab,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        latent_dim=args.latent_dim,
        max_position_embeddings=max_len,
        ffn_mode="dense",
        tie_word_embeddings=True,
    )
    return KilatTransformer(cfg)


def build_llama(args: argparse.Namespace, max_len: int, intermediate: int) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=args.vocab,
        hidden_size=args.d_model,
        intermediate_size=intermediate,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.heads,
        max_position_embeddings=max_len,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    return LlamaForCausalLM(cfg)


def count_params(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for p in model.parameters():  # parameters() already de-duplicates tied weights
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def _meta_llama_params(args: argparse.Namespace, max_len: int, intermediate: int) -> int:
    with torch.device("meta"):
        return count_params(build_llama(args, max_len, intermediate))


def solve_llama_intermediate(args: argparse.Namespace, max_len: int, target: int) -> int:
    """Largest intermediate_size with LLaMA params <= target (params are linear in it)."""
    p0 = _meta_llama_params(args, max_len, 64)
    p1 = _meta_llama_params(args, max_len, 128)
    slope = (p1 - p0) / 64.0
    guess = int((target - p0) / slope) + 64
    best = guess
    for cand in range(guess + 4, guess - 5, -1):
        if _meta_llama_params(args, max_len, cand) <= target:
            best = cand
            break
    return best


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure(model: torch.nn.Module, length: int, args: argparse.Namespace, device: str) -> dict[str, Any]:
    """Time one config. Returns an OOM marker instead of raising."""
    try:
        _cleanup()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        x = torch.randint(100, 1000, (args.batch, length), generator=gen).to(device)
        grad_ctx = torch.enable_grad() if args.train else torch.inference_mode()
        model.train(args.train)
        with grad_ctx:
            for _ in range(args.warmup):
                out = model(x, labels=x) if args.train else model(x)
                if args.train:
                    (out.loss if hasattr(out, "loss") else out[0]).backward()
                    model.zero_grad(set_to_none=True)
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = model(x, labels=x) if args.train else model(x)
                if args.train:
                    (out.loss if hasattr(out, "loss") else out[0]).backward()
                    model.zero_grad(set_to_none=True)
            _sync(device)
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        return {
            "ms": elapsed / args.iters * 1000.0,
            "tok_s": args.batch * length * args.iters / elapsed,
            "peak_mb": peak,
            "oom": False,
        }
    except torch.OutOfMemoryError:
        return {"ms": None, "tok_s": None, "peak_mb": None, "oom": True}
    finally:
        _cleanup()


def _fmt(v: float | None, spec: str, unit: str = "", oom: bool = False) -> str:
    if oom:
        return "OOM"
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--latent-dim", type=int, default=96)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", action="store_true", help="forward+backward instead of inference")
    ap.add_argument("--cpu", action="store_true", help="force CPU/float32 (smoke test)")
    ap.add_argument("--json", type=str, default="fair_scaling_results.json")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32
    max_len = max(args.lengths)
    torch.manual_seed(args.seed)

    kilat = build_kilat(args, max_len)
    p_kilat = count_params(kilat)
    inter = solve_llama_intermediate(args, max_len, p_kilat)
    llama = build_llama(args, max_len, inter)
    p_llama = count_params(llama)
    diff_pct = (p_kilat - p_llama) / p_llama * 100.0

    # ---- Assurance gates -------------------------------------------------------------------
    if p_llama > p_kilat:
        raise RuntimeError(f"LLaMA ({p_llama:,}) must not be heavier than Kilat ({p_kilat:,})")
    if abs(diff_pct) > TOLERANCE_PCT:
        raise RuntimeError(f"Parameter parity violated: {diff_pct:.4f}% > {TOLERANCE_PCT}%")
    kilat = kilat.to(device=device, dtype=dtype)
    llama = llama.to(device=device, dtype=dtype)

    mode = "TRAIN (fwd+bwd)" if args.train else "INFERENCE (prefill)"
    print("=" * 96)
    print(f"FAIR SCALING BENCHMARK | mode={mode} | device={device} | dtype={dtype} | batch={args.batch}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Kilat dense : {p_kilat:>12,} params")
    print(f"LLaMA dense : {p_llama:>12,} params (intermediate_size={inter}, sdpa)")
    print(f"Parity      : Kilat is {diff_pct:+.4f}% vs LLaMA (tolerance {TOLERANCE_PCT}%; Kilat >= LLaMA)")
    print("=" * 96)
    header = f"{'Len':>7} | {'Kilat ms':>10} {'LLaMA ms':>10} {'Time ratio':>10} | {'Kilat MB':>9} {'LLaMA MB':>9} {'VRAM save':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for length in args.lengths:
        rk = measure(kilat, length, args, device)
        rl = measure(llama, length, args, device)
        ratio = rl["ms"] / rk["ms"] if rk["ms"] and rl["ms"] else None  # >1 => Kilat faster
        save = (
            (rl["peak_mb"] - rk["peak_mb"]) / rl["peak_mb"] * 100.0
            if rk["peak_mb"] and rl["peak_mb"]
            else None
        )
        rows.append({"length": length, "kilat": rk, "llama": rl, "llama_over_kilat_time": ratio, "vram_saving_pct": save})
        ko, lo = rk["oom"], rl["oom"]
        print(
            f"{length:>7} | {_fmt(rk['ms'], '.1f', oom=ko):>10} {_fmt(rl['ms'], '.1f', oom=lo):>10} "
            f"{_fmt(ratio, '.2f', 'x'):>10} | {_fmt(rk['peak_mb'], '.0f', oom=ko):>9} "
            f"{_fmt(rl['peak_mb'], '.0f', oom=lo):>9} {_fmt(save, '.1f', '%'):>9}"
        )

    report = {
        "mode": mode,
        "device": device,
        "dtype": str(dtype),
        "params": {"kilat": p_kilat, "llama": p_llama, "diff_pct": diff_pct, "llama_intermediate": inter},
        "protocol": {
            "batch": args.batch, "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "llama_attn": "sdpa", "tied_embeddings": True,
        },
        "rows": rows,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.json}  (Time ratio > 1.00x means Kilat is faster; OOM = out of memory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
