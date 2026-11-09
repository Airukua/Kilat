"""
Multi-Query Associative Recall (MQAR) benchmark for Kilat sequence mixers.

Methodology follows Zoology (Arora et al., ICLR 2024, "Zoology: Measuring and
Improving Recall in Efficient Language Models"):

* Each sequence starts with ``D`` key-value pairs ``k1 v1 k2 v2 ... kD vD``.
* The remaining positions hold the ``D`` keys again (shuffled) at random
  query slots; every other position is a filler token.
* At every query key the model must emit the value that was bound to that key
  earlier in the context. Loss/accuracy is computed only on those positions.
* Small models (2 layers) are swept over sequence length and model width.
  Attention solves MQAR with a small constant width, while fixed-state
  recurrent mixers need width that grows with sequence length.

Kilat-specific adaptation: the model shifts labels internally
(``logits[t]`` predicts ``labels[t+1]``), so the target value is placed at
``labels[p + 1]`` for a query at position ``p``. Training data is generated
fresh every step (no memorisation possible); evaluation uses a fixed held-out
set generated from a different seed.

Usage
-----
    python tools/synthetic/mqar.py --ratios 0.0 0.5 1.0 --seq-lens 64 128 256 \
        --widths 64 128 --steps 3000 --out artifacts/mqar/results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from kilat import KilatConfig, KilatTransformer

FILLER_TOKEN = 0


def make_mqar_batch(
    batch_size: int,
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build one MQAR batch.

    Keys are drawn from ``[1, vocab_size // 2)`` and values from
    ``[vocab_size // 2, vocab_size)`` so the two sets never overlap. Keys within
    a sequence are unique. Query keys sit on even offsets of the query region so
    that position ``p + 1`` (where the label lives) is always a filler token.

    Returns
    -------
    input_ids : (B, N) LongTensor
    labels    : (B, N) LongTensor, ``-100`` everywhere except answer slots.
    """
    d = num_kv_pairs
    ctx = 2 * d
    query_region = seq_len - ctx
    if query_region < 2 * d:
        raise ValueError(
            f"seq_len={seq_len} too short for {d} kv pairs + {d} spaced queries"
        )
    half = vocab_size // 2

    input_ids = torch.full((batch_size, seq_len), FILLER_TOKEN, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # Unique keys per row: argsort of random scores, take first d.
    key_pool = torch.rand(batch_size, half - 1, generator=generator).argsort(dim=1)[:, :d] + 1
    values = torch.randint(half, vocab_size, (batch_size, d), generator=generator)

    input_ids[:, 0:ctx:2] = key_pool
    input_ids[:, 1:ctx:2] = values

    # Query slots: choose d of the available even offsets (needs p+1 < seq_len).
    n_slots = query_region // 2
    slot_scores = torch.rand(batch_size, n_slots, generator=generator)
    slots = slot_scores.argsort(dim=1)[:, :d]  # (B, d)
    query_pos = ctx + 2 * slots  # absolute positions
    order = torch.rand(batch_size, d, generator=generator).argsort(dim=1)  # which key at each slot

    rows = torch.arange(batch_size).unsqueeze(1).expand(-1, d)
    q_keys = torch.gather(key_pool, 1, order)
    q_vals = torch.gather(values, 1, order)
    input_ids[rows, query_pos] = q_keys
    labels[rows, query_pos + 1] = q_vals
    return input_ids, labels


@torch.no_grad()
def evaluate(
    model: KilatTransformer,
    eval_set: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    """Exact-match accuracy over all answer slots of the held-out set."""
    model.eval()
    correct = 0
    total = 0
    for x, y in eval_set:
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x).logits
        pred = logits[:, :-1].argmax(-1)
        tgt = y[:, 1:]
        mask = tgt != -100
        correct += int((pred[mask] == tgt[mask]).sum().item())
        total += int(mask.sum().item())
    model.train()
    return correct / max(1, total)


def run_one(
    recall_ratio: float,
    seq_len: int,
    width: int,
    lr: float,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    num_kv = args.num_kv_pairs or max(4, seq_len // 16)
    cfg = KilatConfig(
        vocab_size=args.vocab_size,
        n_embd=width,
        n_layer=args.n_layer,
        n_head=args.n_head,
        recall_ratio=recall_ratio,
        latent_dim=max(16, width // 2),
        use_rope=True,
        rope_head_dim=8,
        ffn_mode="swiglu",
        max_position_embeddings=seq_len,
        tie_word_embeddings=True,
    )
    model = KilatTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_nonemb = n_params - args.vocab_size * width

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    warmup = max(1, args.steps // 20)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, args.steps - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    g_train = torch.Generator().manual_seed(args.seed + 1)
    g_eval = torch.Generator().manual_seed(10_000 + args.seed)
    eval_set = [
        make_mqar_batch(args.batch_size, seq_len, num_kv, args.vocab_size, g_eval)
        for _ in range(args.eval_batches)
    ]

    best_acc = 0.0
    final_loss = float("nan")
    t0 = time.perf_counter()
    pbar = tqdm(
        range(args.steps),
        desc=f"  ratio={recall_ratio:.1f} N={seq_len} d={width} lr={lr:.0e}",
        leave=False,
        dynamic_ncols=True,
    )
    for step in pbar:
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = make_mqar_batch(args.batch_size, seq_len, num_kv, args.vocab_size, g_train)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(x, labels=y).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            final_loss = float(loss.item())
            acc = evaluate(model, eval_set, device, amp_dtype)
            best_acc = max(best_acc, acc)
            pbar.set_postfix({"loss": f"{final_loss:.3f}", "acc": f"{acc:.3f}", "best": f"{best_acc:.3f}"})
            if best_acc >= 0.999:
                pbar.write(f"    ✓ Solved early at step {step + 1} (acc={best_acc:.3f})")
                break
    pbar.close()

    lam = None
    lam_vals = []
    for mod in model.modules():
        log_lam = getattr(mod, "log_lambda", None)
        if isinstance(log_lam, torch.nn.Parameter):
            lam_vals.extend(torch.sigmoid(log_lam.detach().float()).cpu().tolist())
    if lam_vals:
        lam = {"min": min(lam_vals), "max": max(lam_vals)}

    return {
        "recall_ratio": recall_ratio,
        "seq_len": seq_len,
        "width": width,
        "lr": lr,
        "num_kv_pairs": num_kv,
        "params_total": n_params,
        "params_non_embedding": n_nonemb,
        "best_eval_acc": round(best_acc, 4),
        "final_train_loss": round(final_loss, 4),
        "steps_run": step + 1,
        "seconds": round(time.perf_counter() - t0, 1),
        "decay_lambda": lam,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Zoology-style MQAR sweep for Kilat")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    p.add_argument("--num-kv-pairs", type=int, default=0, help="0 = max(4, seq_len // 16)")
    p.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size (Zoology standard: 256 for d=64)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer (default: 0.0)")
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/mqar/results.json")
    args = p.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = list(itertools.product(args.ratios, args.seq_lens, args.widths))
    print(f"\n⚡ KILAT MQAR Benchmark ({len(configs)} configurations to evaluate)")
    print(f"Device: {device} | Steps/run: {args.steps} | Batch size: {args.batch_size} | LRs: {args.lrs}\n", flush=True)

    results: List[Dict[str, Any]] = []
    for idx, (ratio, seq_len, width) in enumerate(configs):
        num_kv = args.num_kv_pairs or max(4, seq_len // 16)
        print(
            f"[{idx + 1:2d}/{len(configs):2d}] Sweeping ratio={ratio:.1f} | N={seq_len:3d} | d={width:3d} (kv={num_kv})...",
            flush=True,
        )
        # Zoology protocol: sweep LR, report the best run per configuration.
        runs = [run_one(ratio, seq_len, width, lr, args, device) for lr in args.lrs]
        best = max(runs, key=lambda r: r["best_eval_acc"])
        best["lr_sweep"] = {str(r["lr"]): r["best_eval_acc"] for r in runs}
        results.append(best)
        print(
            f"  └─► BEST: acc={best['best_eval_acc']:.3f} | lr={best['lr']:.0e} "
            f"| lambda={best['decay_lambda']} | time={best['seconds']}s",
            flush=True,
        )
        out_path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
