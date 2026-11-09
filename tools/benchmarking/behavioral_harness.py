"""
Behavioral Falsification Benchmark Harness.

Executes controlled counterfactual evaluation pairs:
- Tests Object Tracking across names, colors, and templates.
- Evaluates Logit Margins: M = z_target - z_foil.
- Determines whether model behavior is genuinely causal or template-memorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from kilat.configs.main_config import MainConfig
from kilat.models.transformer import KilatTransformer


def evaluate_counterfactual_case(
    model: KilatTransformer,
    tokenizer: Any,
    prompt: str,
    expected_target: str,
    foil_target: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single prompt against its expected target and foil."""
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)

    # Encode expected and foil tokens (take first subword token)
    expected_ids = tokenizer.encode(expected_target)
    foil_ids = tokenizer.encode(foil_target)

    exp_tok = expected_ids[0]
    foil_tok = foil_ids[0]

    z_target = float(logits[exp_tok].item())
    z_foil = float(logits[foil_tok].item())
    logit_margin = z_target - z_foil

    p_target = float(probs[exp_tok].item())
    p_foil = float(probs[foil_tok].item())

    # Check top-1 prediction
    top_pred_id = int(torch.argmax(logits).item())
    top_pred_text = tokenizer.decode([top_pred_id])
    is_correct = (top_pred_id == exp_tok) or (logit_margin > 0)

    return {
        "prompt": prompt,
        "expected": expected_target,
        "foil": foil_target,
        "logit_margin": round(logit_margin, 4),
        "target_prob": round(p_target * 100, 2),
        "foil_prob": round(p_foil * 100, 2),
        "top1_predicted": repr(top_pred_text),
        "is_correct": bool(is_correct),
    }


def run_behavioral_suite(
    model_dir: Union[str, Path],
    test_cases_path: Union[str, Path],
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Run full behavioral falsification suite."""
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu"
    )

    ckpt_p = Path(model_dir)
    model = KilatTransformer.from_pretrained(str(ckpt_p)).to(device).eval()

    cfg_yaml = ckpt_p / "config.yaml"
    if cfg_yaml.is_file():
        cfg = MainConfig.from_yaml(cfg_yaml)
        tokenizer = cfg.tokenizer.build()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_p), local_files_only=True)

    tc_path = Path(test_cases_path)
    with tc_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[Dict[str, Any]] = []
    correct_count = 0
    total_margin = 0.0

    category_stats: Dict[str, Dict[str, Any]] = {}

    for c in cases:
        cat = c.get("category", "general")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "margins": []}

        outcome = evaluate_counterfactual_case(
            model=model,
            tokenizer=tokenizer,
            prompt=c["prompt"],
            expected_target=c["expected_target"],
            foil_target=c["foil_target"],
            device=device,
        )

        outcome["case_id"] = c.get("case_id", "unnamed")
        outcome["category"] = cat
        outcome["description"] = c.get("description", "")

        results.append(outcome)

        category_stats[cat]["total"] += 1
        category_stats[cat]["margins"].append(outcome["logit_margin"])
        if outcome["is_correct"]:
            correct_count += 1
            category_stats[cat]["correct"] += 1

        total_margin += outcome["logit_margin"]

    overall_accuracy = correct_count / max(1, len(cases))
    mean_margin = total_margin / max(1, len(cases))

    # Summarize categories
    cat_summary = {}
    for cat, st in category_stats.items():
        tot = st["total"]
        cor = st["correct"]
        cat_summary[cat] = {
            "accuracy": round(cor / max(1, tot) * 100, 1),
            "mean_logit_margin": round(sum(st["margins"]) / max(1, tot), 3),
            "passed": cor,
            "total": tot,
        }

    return {
        "model_checkpoint": str(ckpt_p),
        "total_cases": len(cases),
        "passed_cases": correct_count,
        "accuracy_pct": round(overall_accuracy * 100, 2),
        "mean_logit_margin": round(mean_margin, 4),
        "category_summary": cat_summary,
        "details": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Kilat Behavioral Falsification Harness")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("-t", "--test-cases", type=str, default="tools/benchmarking/test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save report as JSON")
    args = parser.parse_args()

    report = run_behavioral_suite(
        model_dir=args.checkpoint,
        test_cases_path=args.test_cases,
        device_str=args.device,
    )

    print(json.dumps(report, indent=2))
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved to: {out_p}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
