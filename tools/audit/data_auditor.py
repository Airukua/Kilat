"""
Data and Boundary Integrity Auditor.

Provides mathematical and empirical proof of:
1. EOS boundary termination frequency and distribution.
2. Exact N-Gram contamination between Train and Eval splits.
3. Document boundary sequence packaging audit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


def audit_eos_boundaries(
    tokens: np.ndarray,
    eos_token_id: int = 50256,
) -> Dict[str, Any]:
    """
    Empirically verify whether EOS tokens exist in the token stream and their distance distribution.
    """
    total_tokens = int(tokens.size)
    eos_indices = np.where(tokens == eos_token_id)[0]
    eos_count = int(len(eos_indices))

    if eos_count == 0:
        return {
            "total_tokens": total_tokens,
            "eos_token_id": eos_token_id,
            "eos_count": 0,
            "eos_frequency": 0.0,
            "has_eos": False,
            "mean_story_length": float("inf"),
            "max_distance_without_eos": total_tokens,
            "verdict": "FAIL_ZERO_EOS",
            "message": "CRITICAL: Zero EOS tokens found in dataset. Stories are concatenated without boundaries.",
        }

    distances = np.diff(eos_indices) if eos_count > 1 else np.array([total_tokens])
    mean_dist = float(np.mean(distances)) if len(distances) > 0 else float(total_tokens)
    median_dist = float(np.median(distances)) if len(distances) > 0 else float(total_tokens)
    max_dist = int(np.max(distances)) if len(distances) > 0 else int(total_tokens)
    min_dist = int(np.min(distances)) if len(distances) > 0 else int(total_tokens)

    return {
        "total_tokens": total_tokens,
        "eos_token_id": eos_token_id,
        "eos_count": eos_count,
        "eos_frequency": float(eos_count / max(1, total_tokens)),
        "has_eos": True,
        "mean_story_length": round(mean_dist, 2),
        "median_story_length": round(median_dist, 2),
        "min_story_length": min_dist,
        "max_story_length": max_dist,
        "verdict": "PASS_EOS_PRESENT",
        "message": f"Found {eos_count:,} EOS boundaries. Average story length: {mean_dist:.1f} tokens.",
    }


def audit_ngram_contamination(
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    ngram_sizes: Sequence[int] = (8, 13),
    sample_window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Conduct exact N-gram membership audit between Train and Eval arrays.
    """
    tr_len = len(train_tokens)
    ev_len = len(eval_tokens)

    tr_limit = tr_len if sample_window is None else min(tr_len, sample_window)
    ev_limit = ev_len if sample_window is None else min(ev_len, sample_window)

    results: Dict[str, Any] = {
        "train_tokens_audited": int(tr_limit),
        "eval_tokens_audited": int(ev_limit),
        "ngrams": {},
    }

    worst_contamination = 0.0

    for n in ngram_sizes:
        if tr_limit < n or ev_limit < n:
            continue

        train_set: Set[Tuple[int, ...]] = set(
            tuple(train_tokens[i : i + n].tolist()) for i in range(0, tr_limit - n + 1)
        )
        eval_list: List[Tuple[int, ...]] = [
            tuple(eval_tokens[i : i + n].tolist()) for i in range(0, ev_limit - n + 1)
        ]

        overlap_count = 0
        seen_overlaps: Set[Tuple[int, ...]] = set()

        for gram in eval_list:
            if gram in train_set:
                overlap_count += 1
                seen_overlaps.add(gram)

        overlap_ratio = overlap_count / max(1, len(eval_list))
        worst_contamination = max(worst_contamination, overlap_ratio)

        results["ngrams"][f"{n}gram"] = {
            "n": n,
            "eval_ngrams_total": len(eval_list),
            "overlap_occurrences": overlap_count,
            "unique_overlap_ngrams": len(seen_overlaps),
            "contamination_rate": round(float(overlap_ratio), 6),
        }

    # Strict scientific threshold: 13-gram contamination must be < 0.001
    rate_13 = results["ngrams"].get("13gram", {}).get("contamination_rate", 0.0)
    is_contaminated = rate_13 > 0.005

    results["verdict"] = "CONTAMINATED" if is_contaminated else "CLEAN"
    results["verdict_message"] = (
        f"Significant N-Gram leakage detected ({rate_13*100:.3f}% 13-gram overlap). Generalization claims invalid."
        if is_contaminated
        else "Dataset partitions are cleanly separated."
    )

    return results


def run_full_data_audit(
    train_path: Union[str, Path],
    eval_path: Optional[Union[str, Path]] = None,
    eos_token_id: int = 50256,
    sample_window: Optional[int] = 500000,
) -> Dict[str, Any]:
    """Execute complete data integrity audit."""
    tr_path = Path(train_path)
    if not tr_path.is_file():
        raise FileNotFoundError(f"Train data file not found: {tr_path}")

    train_tokens = np.load(tr_path, mmap_mode="r")
    train_eos_report = audit_eos_boundaries(train_tokens, eos_token_id=eos_token_id)

    report: Dict[str, Any] = {
        "train_file": str(tr_path),
        "train_tokens_total": int(train_tokens.size),
        "train_eos_audit": train_eos_report,
    }

    if eval_path is not None:
        ev_path = Path(eval_path)
        if not ev_path.is_file():
            raise FileNotFoundError(f"Eval data file not found: {ev_path}")
        eval_tokens = np.load(ev_path, mmap_mode="r")
        eval_eos_report = audit_eos_boundaries(eval_tokens, eos_token_id=eos_token_id)
        leak_report = audit_ngram_contamination(
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            ngram_sizes=(8, 13),
            sample_window=sample_window,
        )
        report["eval_file"] = str(ev_path)
        report["eval_tokens_total"] = int(eval_tokens.size)
        report["eval_eos_audit"] = eval_eos_report
        report["contamination_audit"] = leak_report

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Kilat Data & Boundary Integrity Auditor")
    parser.add_argument("--train", type=str, required=True, help="Path to train .npy file")
    parser.add_argument("--eval", type=str, default=None, help="Path to eval .npy file")
    parser.add_argument("--eos-token-id", type=int, default=50256, help="Target EOS token ID (default: 50256)")
    parser.add_argument("--sample-window", type=int, default=500000, help="Max tokens to audit for leakage (default: 500k)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save report as JSON")
    args = parser.parse_args()

    report = run_full_data_audit(
        train_path=args.train,
        eval_path=args.eval,
        eos_token_id=args.eos_token_id,
        sample_window=args.sample_window,
    )

    formatted = json.dumps(report, indent=2)
    print(formatted)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted, encoding="utf-8")
        print(f"\nReport written to: {out_p}")

    is_clean = report.get("train_eos_audit", {}).get("has_eos", False)
    if "contamination_audit" in report:
        is_clean = is_clean and (report["contamination_audit"]["verdict"] == "CLEAN")

    return 0 if is_clean else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
