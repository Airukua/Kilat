"""
Per-position validation loss: does the model actually use long context?

Standard long-context diagnostic (used in e.g. the Gemini 1.5 report and in
linear-attention / SSM papers): plot the mean next-token cross-entropy as a
function of how much in-document context precedes the target token. A model
that exploits context keeps lowering its loss as context grows; a model whose
effective memory is short flattens early.

Two axes are reported:
* ``doc_pos``    — tokens since the last EOS (position inside the story).
* ``window_pos`` — absolute position inside the evaluation window.

Usage
-----
    python tools/analysis/position_loss.py \
        -c checkpoints/run_a/checkpoint-best checkpoints/run_b/checkpoint-best \
        -d experiments/eval.npy --num-windows 256 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from kilat.models.transformer import KilatTransformer

BUCKETS = [0, 8, 16, 32, 64, 128, 256, 512, 1_000_000]


def _bucket_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi - 1}" if hi < 1_000_000 else f"{lo}+"


@torch.no_grad()
def position_loss(
    ckpt: str,
    data_path: str,
    seq_len: int,
    num_windows: int,
    batch_size: int,
    eos_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    model = KilatTransformer.from_pretrained(ckpt).to(device).eval()
    tokens = np.load(data_path, mmap_mode="r")
    n_avail = (len(tokens) - 1) // seq_len
    n_win = min(num_windows, n_avail)
    # Evenly spaced windows -> deterministic and covers the whole split.
    starts = np.linspace(0, (n_avail - 1) * seq_len, n_win).astype(np.int64)

    n_b = len(BUCKETS) - 1
    doc_sum = np.zeros(n_b)
    doc_cnt = np.zeros(n_b)
    win_sum = np.zeros(n_b)
    win_cnt = np.zeros(n_b)
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    for i in range(0, n_win, batch_size):
        chunk = np.stack([np.asarray(tokens[s : s + seq_len + 1]) for s in starts[i : i + batch_size]])
        ids = torch.from_numpy(chunk.astype(np.int64))
        x = ids[:, :-1].to(device)
        y = ids[:, 1:].to(device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(x).logits
        ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").cpu().numpy()

        # Tokens of in-document context available when predicting y[:, t]:
        # count of tokens in x[:, :t+1] after the last EOS (EOS itself excluded).
        x_np = x.cpu().numpy()
        doc_pos = np.zeros_like(x_np)
        run = np.zeros(x_np.shape[0], dtype=np.int64)
        for t in range(x_np.shape[1]):
            is_eos = x_np[:, t] == eos_id
            run = np.where(is_eos, 0, run + 1)
            doc_pos[:, t] = run
        target_is_eos = y.cpu().numpy() == eos_id  # skip predicting boundaries
        win_pos = np.broadcast_to(np.arange(x_np.shape[1]) + 1, x_np.shape)

        for b in range(n_b):
            lo, hi = BUCKETS[b], BUCKETS[b + 1]
            m = (doc_pos >= lo) & (doc_pos < hi) & (doc_pos > 0) & ~target_is_eos
            doc_sum[b] += ce[m].sum()
            doc_cnt[b] += m.sum()
            m2 = (win_pos >= lo) & (win_pos < hi) & ~target_is_eos
            win_sum[b] += ce[m2].sum()
            win_cnt[b] += m2.sum()

    def table(s: np.ndarray, c: np.ndarray) -> List[Dict[str, Any]]:
        return [
            {"bucket": _bucket_label(BUCKETS[b], BUCKETS[b + 1]), "tokens": int(c[b]),
             "mean_ce": round(float(s[b] / c[b]), 4)}
            for b in range(n_b) if c[b] > 0
        ]

    return {
        "checkpoint": ckpt,
        "windows": int(n_win),
        "overall_ce": round(float(doc_sum.sum() / max(1, doc_cnt.sum())), 4),
        "by_doc_position": table(doc_sum, doc_cnt),
        "by_window_position": table(win_sum, win_cnt),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Per-position validation loss")
    p.add_argument("-c", "--checkpoints", nargs="+", required=True)
    p.add_argument("-d", "--data", default="experiments/eval.npy")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-windows", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eos-id", type=int, default=50256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    device = torch.device(args.device)
    reports = [
        position_loss(c, args.data, args.seq_len, args.num_windows, args.batch_size, args.eos_id, device)
        for c in args.checkpoints
    ]
    for r in reports:
        print(f"\n== {r['checkpoint']}  (overall CE {r['overall_ce']}, {r['windows']} windows)")
        print(f"{'doc position':>14} | {'tokens':>8} | mean CE")
        for row in r["by_doc_position"]:
            print(f"{row['bucket']:>14} | {row['tokens']:>8} | {row['mean_ce']:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(reports, indent=2))
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
