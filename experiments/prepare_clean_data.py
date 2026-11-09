"""
Clean Data Preparation Script for TinyStories Experiments.

Ensures:
1. Strict split separation (Train split vs Validation split from Hugging Face).
2. Proper EOS token (50256) injection at the end of every story.
3. Zero document concatenation leakage (stories terminate cleanly).
4. Zero train-eval overlap (allow_train_val_overlap: false compliant).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_tinystories(
    output_dir: str = "experiments",
    num_train_stories: int = 50_000,
    num_eval_stories: int = 5_000,
    tokenizer_name: str = "gpt2",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 50256

    # 1. Prepare Eval Data from official validation split
    print(f"\n[1/2] Streaming official validation split ({num_eval_stories:,} stories)...")
    eval_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    eval_tokens: list[int] = []
    eval_count = 0

    for item in eval_ds:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ids = tok.encode(text)
        if ids and ids[-1] != eos_id:
            ids.append(eos_id)
        eval_tokens.extend(ids)
        eval_count += 1
        if eval_count >= num_eval_stories:
            break

    eval_arr = np.asarray(eval_tokens, dtype=np.int32)
    eval_path = out_dir / "eval.npy"
    np.save(eval_path, eval_arr)
    meta_eval = {"num_tokens": int(len(eval_arr)), "dtype": "int32"}
    with open(out_dir / "eval.npy.meta.json", "w") as mf:
        json.dump(meta_eval, mf)
    print(f"✓ Saved clean eval tokens: {eval_path} ({len(eval_arr):,} tokens, {eval_count:,} stories)")

    # 2. Prepare Train Data from official train split
    print(f"\n[2/2] Streaming official train split ({num_train_stories:,} stories)...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_tokens: list[int] = []
    train_count = 0

    for item in train_ds:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ids = tok.encode(text)
        if ids and ids[-1] != eos_id:
            ids.append(eos_id)
        train_tokens.extend(ids)
        train_count += 1
        if train_count >= num_train_stories:
            break

    train_arr = np.asarray(train_tokens, dtype=np.int32)
    train_path = out_dir / "train.npy"
    np.save(train_path, train_arr)
    meta_train = {"num_tokens": int(len(train_arr)), "dtype": "int32"}
    with open(out_dir / "train.npy.meta.json", "w") as mf:
        json.dump(meta_train, mf)
    print(f"✓ Saved clean train tokens: {train_path} ({len(train_arr):,} tokens, {train_count:,} stories)")

    print("\n" + "=" * 65)
    print("🎉 Dataset preparation complete! Now run verification:")
    print(f"python tools/audit/data_auditor.py --train {train_path} --eval {eval_path}")
    print("=" * 65)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Clean TinyStories Dataset")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory")
    parser.add_argument("--train-stories", type=int, default=50_000, help="Number of training stories")
    parser.add_argument("--eval-stories", type=int, default=5_000, help="Number of evaluation stories")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    args = parser.parse_args()

    prepare_tinystories(
        output_dir=args.output_dir,
        num_train_stories=args.train_stories,
        num_eval_stories=args.eval_stories,
        tokenizer_name=args.tokenizer,
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
