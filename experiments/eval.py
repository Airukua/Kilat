"""
Evaluation harness and benchmark runner for TinyStories language models.

Based on the empirical methodology of:
"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
(Ronen Eldan & Yuanzhi Li, Microsoft Research, 2023).

Evaluates:
1. Exact Validation Cross-Entropy Loss & Perplexity (token-weighted NLL).
2. Story continuation quality across canonical TinyStories evaluation prompts.
3. Degeneracy metrics (repetition ratios, distinct n-grams).
4. Generates structured JSON and Markdown report.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import yaml

from kilat.configs.main_config import MainConfig
from kilat.data.collator import KilatDataCollator
from kilat.data.dataset import PretrainingDataset
from kilat.inference.generation_config import GenerationConfig
from kilat.inference.generator import TextGenerator
from kilat.models.transformer import KilatTransformer


def load_tinystories_prompts(prompts_path: Union[str, Path]) -> List[str]:
    """Load canonical prompts from YAML file."""
    path = Path(prompts_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation prompts file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list of prompts in {path}, got {type(data)}")

    prompts = [str(p).strip() for p in data if str(p).strip()]
    return prompts


def compute_ngram_repetition_rate(tokens: Sequence[int], n: int = 4) -> float:
    """Compute ratio of duplicate n-grams in generated sequence."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    unique_count = len(set(ngrams))
    return 1.0 - (unique_count / len(ngrams))


def evaluate_dataset_perplexity(
    model: KilatTransformer,
    eval_data_path: Union[str, Path],
    max_seq_len: int = 512,
    batch_size: int = 8,
    max_batches: Optional[int] = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute exact token-weighted cross-entropy loss and perplexity on validation split."""
    if device is None:
        device = next(model.parameters()).device

    dataset = PretrainingDataset(eval_data_path, chunk_size=max_seq_len)
    collator = KilatDataCollator(pad_token_id=0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, labels=labels)
            # ce_loss is per-token cross entropy
            ce_loss = getattr(model, "_last_loss_breakdown", {}).get("ce_loss", out.loss)
            valid_tokens = int((labels[:, 1:] != -100).sum().item())

            if valid_tokens > 0 and ce_loss is not None:
                total_nll += float(ce_loss.item()) * valid_tokens
                total_tokens += valid_tokens

    if total_tokens == 0:
        return {"eval_loss": float("nan"), "perplexity": float("nan"), "eval_tokens": 0}

    mean_loss = total_nll / total_tokens
    try:
        ppl = math.exp(min(mean_loss, 20.0))
    except OverflowError:
        ppl = float("inf")

    return {
        "eval_loss": float(mean_loss),
        "perplexity": float(ppl),
        "eval_tokens": int(total_tokens),
    }


def run_tinystories_benchmark(
    model_or_checkpoint: Union[KilatTransformer, str, Path],
    eval_prompts_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    eval_data_path: Optional[Union[str, Path]] = None,
    max_prompts: Optional[int] = 10,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    repetition_penalty: float = 1.15,
    do_sample: bool = False,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Run full TinyStories evaluation protocol.

    Args:
        model_or_checkpoint: KilatTransformer instance or path to checkpoint dir.
        eval_prompts_path: Path to official Evaluation prompts.yaml.
        config_path: Optional path to config.yaml if checkpoint lacks it.
        eval_data_path: Optional path to validation .npy for perplexity calculation.
        max_prompts: Maximum number of prompts to evaluate (None for all 44).
        max_new_tokens: Number of tokens to generate per prompt.
        temperature: Sampling temperature (ignored if do_sample=False).
        repetition_penalty: Repetition penalty coefficient.
        do_sample: Whether to use stochastic sampling or greedy decoding.
        device_str: Device identifier ('auto', 'cuda', 'cpu').
    """
    # 1. Device resolution
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 2. Model & Tokenizer resolution
    if isinstance(model_or_checkpoint, (str, Path)):
        ckpt_dir = Path(model_or_checkpoint)
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        model = KilatTransformer.from_pretrained(str(ckpt_dir))
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = model.wte.weight
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(ckpt_dir), local_files_only=True
            )
            if (
                getattr(tokenizer, "eos_token_id", None) is None
                and getattr(model.config, "eos_token_id", None) is not None
            ):
                tokenizer.eos_token_id = int(model.config.eos_token_id)
        except Exception:
            cfg_file = ckpt_dir / "config.yaml"
            if not cfg_file.is_file() and config_path:
                cfg_file = Path(config_path)
            if cfg_file.is_file():
                cfg = MainConfig.from_yaml(cfg_file)
                tokenizer = cfg.tokenizer.build()
            else:
                from kilat.data.tokenizer import ByteLevelDebugTokenizer

                tokenizer = ByteLevelDebugTokenizer(
                    vocab_size=int(model.config.vocab_size)
                )
    else:
        model = model_or_checkpoint
        tokenizer = None

    model = model.to(device)
    model.eval()

    # 3. Validation set perplexity (if data provided)
    ppl_metrics = {}
    if eval_data_path and Path(eval_data_path).is_file():
        ppl_metrics = evaluate_dataset_perplexity(
            model=model,
            eval_data_path=eval_data_path,
            device=device,
        )

    # 4. Prompt evaluation
    prompts = load_tinystories_prompts(eval_prompts_path)
    if max_prompts is not None and max_prompts > 0:
        prompts = prompts[:max_prompts]

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        pad_token_id=0,
        eos_token_id=getattr(tokenizer, "eos_token_id", None) or 50256,
    )

    results: List[Dict[str, Any]] = []
    total_tokens_gen = 0
    start_time = time.perf_counter()

    for idx, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        gen_text = generator.generate(prompt=prompt, generation_config=gen_config)
        elapsed = time.perf_counter() - t0

        # Encode generated tokens for repetition analysis
        if hasattr(generator.tokenizer, "encode"):
            encoded = generator.tokenizer.encode(gen_text)
            token_ids = encoded if isinstance(encoded, list) else encoded.tolist()
        else:
            token_ids = generator._encode_prompt(gen_text)[0].tolist()
        rep_ratio = compute_ngram_repetition_rate(token_ids, n=4)

        results.append({
            "prompt_index": idx + 1,
            "prompt": prompt,
            "completion": gen_text[len(prompt) :].strip(),
            "full_text": gen_text,
            "token_count": len(token_ids),
            "repetition_4gram_rate": float(rep_ratio),
            "generation_time_s": float(elapsed),
        })
        total_tokens_gen += len(token_ids)

    total_time = time.perf_counter() - start_time
    avg_rep_rate = (
        sum(r["repetition_4gram_rate"] for r in results) / len(results)
        if results
        else 0.0
    )

    summary: Dict[str, Any] = {
        "device": str(device),
        "prompts_evaluated": len(results),
        "total_tokens_generated": total_tokens_gen,
        "total_time_s": float(total_time),
        "tokens_per_second": float(total_tokens_gen / max(total_time, 1e-4)),
        "mean_repetition_4gram_rate": float(avg_rep_rate),
        **ppl_metrics,
        "samples": results,
    }

    return summary


def format_tinystories_markdown_report(summary: Dict[str, Any]) -> str:
    """Format evaluation benchmark summary into clean GitHub-flavored markdown."""
    lines = [
        "# TinyStories Empirical Evaluation Benchmark Report",
        "",
        "## 1. Quantitative Metrics",
        "",
        f"- **Device**: `{summary.get('device', 'unknown')}`",
        f"- **Prompts Evaluated**: {summary.get('prompts_evaluated', 0)}",
        f"- **Validation Loss**: {summary.get('eval_loss', 'N/A')}",
        f"- **Validation Perplexity (PPL)**: {summary.get('perplexity', 'N/A')}",
        f"- **Mean 4-Gram Repetition Rate**: {summary.get('mean_repetition_4gram_rate', 0.0):.4f} *(lower is better, < 0.15 indicates healthy non-repetitive text)*",
        f"- **Generation Speed**: {summary.get('tokens_per_second', 0.0):.1f} tokens/sec",
        "",
        "## 2. Qualitative Sample Continuations",
        "",
    ]

    for item in summary.get("samples", []):
        lines.append(f"### Prompt #{item['prompt_index']}")
        lines.append(f"> **Prompt**: *{item['prompt']}*")
        lines.append("")
        lines.append(f"**Generated Continuation**:\n\n{item['completion']}")
        lines.append("")
        lines.append(
            f"*Metrics: {item['token_count']} tokens | 4-gram rep: {item['repetition_4gram_rate']:.3f} | {item['generation_time_s']:.2f}s*"
        )
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="TinyStories Benchmark Evaluation Runner (Independent Experiment Script)"
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument(
        "-p",
        "--prompts",
        type=str,
        default="experiments/eval_prompts.yaml",
        help="Path to TinyStories evaluation prompts YAML (default: experiments/eval_prompts.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Explicit model config YAML if not in checkpoint",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to validation .npy data to compute exact validation loss & perplexity",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=10,
        help="Max prompts to evaluate (default: 10, set 0 or -1 for all 44)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Repetition penalty parameter (default: 1.15)",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic sampling (default: greedy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "-o",
        "--output-report",
        type=str,
        default=None,
        help="Path to save evaluation report (.md or .json)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON output to stdout"
    )

    args = parser.parse_args()

    summary = run_tinystories_benchmark(
        model_or_checkpoint=args.checkpoint,
        eval_prompts_path=args.prompts,
        config_path=args.config,
        eval_data_path=args.eval_data,
        max_prompts=args.max_prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        device_str=args.device,
    )

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.suffix == ".json":
            report_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        else:
            report_path.write_text(
                format_tinystories_markdown_report(summary), encoding="utf-8"
            )
        print(f"Report saved to: {report_path}\n")

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(format_tinystories_markdown_report(summary))

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
