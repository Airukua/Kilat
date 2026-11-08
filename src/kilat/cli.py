"""
Unified Command-Line Interface (`kilat`) for Kilat LLM Pre-Training Framework.

Subcommands:
    - `kilat verify`: Run the 6-Gate Pre-Training Assurance Pipeline (+ Stage 7-8 with `--full`)
    - `kilat chaos`: Run the Active ML Chaos Engineering `Saboteur` suite (`C-01`..`C-08`)
    - `kilat train`: Launch deterministic pre-training from YAML + `--set` dot-overrides
    - `kilat prepare-data`: Convert Parquet/JSONL/TXT corpora to zero-copy `np.memmap` + `.meta.json`
    - `kilat generate`: Run incremental `HybridCache` text generation from a checkpoint
    - `kilat export-hf`: Export a Kilat checkpoint to Hugging Face / safetensors format
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from kilat.assurance.chaos import run_chaos_suite
from kilat.assurance.flight_recorder import FlightRecorderCallback
from kilat.assurance.gates.gate1_theory import run_gate1_theory
from kilat.assurance.runner import run_verification_pipeline
from kilat.configs.main_config import MainConfig
from kilat.data.dataloader import build_dataloaders_from_config
from kilat.data.preprocess import prepare_memmap_dataset
from kilat.data.tokenizer import AutoTokenizer, ByteLevelDebugTokenizer
from kilat.engine.checkpoint import set_seed, write_checkpoint_manifest
from kilat.engine.distributed import (
    cleanup_distributed,
    get_local_rank,
    is_distributed_initialized,
    is_main_process,
    setup_distributed,
)
from kilat.engine.trainer import KilatTrainer
from kilat.inference.export_hf import convert_checkpoint_to_hf
from kilat.inference.generator import TextGenerator
from kilat.models.transformer import KilatTransformer
from kilat.utils.banner import print_kilat_banner

CLI_VERSION = "2.0.0"


def _cmd_verify(args: argparse.Namespace) -> int:
    print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or args.json))
    cfg = MainConfig.from_yaml(args.config)
    if args.set:
        cfg.apply_overrides(args.set)

    report = run_verification_pipeline(
        config=cfg,
        output_dir=args.output_dir,
        full=bool(args.full),
        keep_going=bool(args.keep_going),
        memmap_path=args.memmap_path,
        memmap_dtype=args.memmap_dtype,
        allow_train_val_overlap=bool(getattr(args, "allow_train_val_overlap", False)),
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(f"Assurance Overall Status : {report.overall_status}")
        print(f"Configuration SHA-256    : {report.config_sha256}")
        for g in report.gates:
            badge = "PASS" if g.status == "PASS" else g.status
            print(f"  [{badge:4s}] {g.gate_id:<22s} ({g.duration_ms / 1000.0:.3f}s)")
        print(f"Report written to        : {Path(args.output_dir) / 'assurance_report.json'}")

    return 0 if report.overall_status == "PASS" else 1


def _cmd_chaos(args: argparse.Namespace) -> int:
    print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or args.json))
    set_seed(int(args.seed))
    suite = run_chaos_suite(scenarios=(args.scenarios or "all").strip())
    payload = suite.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.quiet:
        rate_pct = (
            100.0 * suite.scenarios_caught / max(1, suite.scenarios_total)
        )
        print(
            f"Chaos Suite Catch Rate : {suite.scenarios_caught}/{suite.scenarios_total} "
            f"({rate_pct:.1f}%)"
        )
        print(f"Baseline Clean Pass    : {suite.baseline_passed}")
        print(f"Post-Chaos Restore Pass: {suite.post_clean_passed}")
        for r in suite.scenarios:
            mark = "CAUGHT" if r.caught else "MISSED"
            print(
                f"  [{mark:6s}] {r.scenario_id} {r.name:<48s} "
                f"-> {r.observed_code} ({r.duration_ms / 1000.0:.3f}s)"
            )

    return 0 if suite.passed else 1


def _enrich_checkpoint_metadata(ckpt_dir: Path, cfg: MainConfig) -> None:
    if not ckpt_dir.is_dir():
        return
    cfg.to_yaml(ckpt_dir / "config.yaml")
    cfg.tokenizer.to_json(ckpt_dir / "tokenizer_config.json")
    write_checkpoint_manifest(ckpt_dir)


def _cmd_train(args: argparse.Namespace) -> int:
    setup_distributed()
    try:
        is_main = is_main_process()
        print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or not is_main))
        cfg = MainConfig.from_yaml(args.config)
        if args.set:
            cfg.apply_overrides(args.set)

        if is_distributed_initialized() and torch.cuda.is_available():
            device = torch.device(f"cuda:{get_local_rank()}")
            torch.cuda.set_device(device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        set_seed(int(cfg.training.seed))
        train_loader, eval_loader = build_dataloaders_from_config(cfg)
        model = KilatTransformer(cfg.model)

        if not args.skip_preflight:
            g1 = run_gate1_theory(
                model_config=cfg.model,
                model=model,
                weight_decay=float(cfg.training.weight_decay),
            )
            if g1.status != "PASS":
                if is_main:
                    sys.stderr.write(
                        f"[KILAT PREFLIGHT FAILED] {g1.failure_code}: {g1.failure_message}\n"
                    )
                return 1
            if not args.quiet and is_main:
                print(f"Preflight Gate 1 PASS ({g1.duration_ms / 1000.0:.3f}s)")
            # Re-seed so model & dataloader state starts cleanly after preflight probe
            set_seed(int(cfg.training.seed))
            train_loader, eval_loader = build_dataloaders_from_config(cfg)
            model = KilatTransformer(cfg.model)

        model = model.to(device)

        try:
            tokenizer: Any = cfg.tokenizer.build()
        except Exception:
            tokenizer = ByteLevelDebugTokenizer(vocab_size=int(cfg.model.vocab_size))

        train_args = cfg.to_training_arguments()
        flight_recorder = FlightRecorderCallback(output_dir=train_args.output_dir)

        trainer = KilatTrainer(
            model=model,
            args=train_args,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            tokenizer=tokenizer,
            callbacks=[flight_recorder],
        )
        state = trainer.train()

        if is_main:
            out_root = Path(train_args.output_dir)
            if out_root.is_dir():
                cfg.to_yaml(out_root / "config.yaml")
                for ckpt_sub in sorted(out_root.glob("checkpoint-*")):
                    _enrich_checkpoint_metadata(ckpt_sub, cfg)

            if not args.quiet:
                print(
                    f"Training finished at step {state.global_step} "
                    f"(checkpoints in '{train_args.output_dir}')."
                )
        return 0
    finally:
        cleanup_distributed()


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or args.json))
    meta = prepare_memmap_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer=args.tokenizer,
        key_name=args.key_name,
        text_key=args.text_key,
        dtype=np.dtype(args.dtype),
        force_overwrite=bool(args.force),
        verbose=not bool(args.quiet or args.json),
    )
    if args.json:
        print(json.dumps(meta, indent=2))
    elif not args.quiet:
        print(
            f"Prepared memmap: {meta['output_path']} "
            f"({meta['num_tokens']} tokens, dtype={meta['dtype']}, sha256={str(meta['sha256'])[:12]}...)"
        )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or args.json))
    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_dir():
        sys.stderr.write(f"Checkpoint directory not found: {ckpt_dir}\n")
        return 1

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    model = KilatTransformer.from_pretrained(str(ckpt_dir))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
        if getattr(tokenizer, "eos_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
            tokenizer.eos_token_id = int(model.config.eos_token_id)
    except Exception:
        tokenizer = ByteLevelDebugTokenizer(vocab_size=int(model.config.vocab_size))

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device_str)
    generated = generator.generate(
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        repetition_penalty=float(getattr(args, "repetition_penalty", 1.0)),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": str(ckpt_dir),
                    "prompt": args.prompt,
                    "generated_text": generated,
                },
                indent=2,
            )
        )
    else:
        print(generated)
    return 0


def _cmd_export_hf(args: argparse.Namespace) -> int:
    print_kilat_banner(version=CLI_VERSION, quiet=bool(args.quiet or args.json))
    results = convert_checkpoint_to_hf(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output,
        config_path=args.config,
        use_safetensors=not bool(args.no_safetensors),
        device=args.device,
        strict=bool(args.strict),
        skip_validation=bool(args.skip_validation),
        skip_tokenizer=bool(args.skip_tokenizer),
        verbose=not bool(args.quiet or args.json),
    )
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    elif not args.quiet:
        print(f"Exported Hugging Face checkpoint to: {results['output_dir']}")
    return 0 if results.get("success") else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``kilat`` argument parser and subcommands."""
    parser = argparse.ArgumentParser(
        prog="kilat",
        description="Kilat — Research-Grade Hybrid Sub-Quadratic & MoE LLM Pre-Training CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"kilat {CLI_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. kilat verify
    p_verify = subparsers.add_parser(
        "verify",
        help="Run the 6-Gate Pre-Training Assurance Pipeline and emit assurance_report.{json,md}",
    )
    p_verify.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/debug_cpu.yaml",
        help="Path to MainConfig YAML file (default: configs/debug_cpu.yaml)",
    )
    p_verify.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./artifacts/verify",
        help="Directory to write assurance_report.json and assurance_report.md",
    )
    p_verify.add_argument(
        "--full",
        action="store_true",
        help="Also execute Stage 7 WSD/muP scaling checks and Stage 8 readiness sign-off",
    )
    p_verify.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running subsequent gates even if an earlier gate fails",
    )
    p_verify.add_argument(
        "--memmap-path",
        type=str,
        default=None,
        help="Optional binary/npy memmap path for Gate 4 on-disk integrity check",
    )
    p_verify.add_argument(
        "--memmap-dtype",
        type=str,
        default="uint16",
        help="NumPy dtype of --memmap-path (default: uint16)",
    )
    p_verify.add_argument(
        "--allow-train-val-overlap",
        action="store_true",
        help="Permit train and validation 13-gram overlap (for natural corpora like TinyStories)",
    )
    p_verify.add_argument(
        "--set",
        "--override",
        dest="set",
        action="append",
        default=[],
        help="Dot-notation config override (e.g. --override training.learning_rate=1e-4)",
    )
    p_verify.add_argument("-q", "--quiet", action="store_true", help="Suppress banner and progress")
    p_verify.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
    p_verify.set_defaults(handler=_cmd_verify)

    # 2. kilat chaos
    p_chaos = subparsers.add_parser(
        "chaos",
        help="Run Active ML Chaos Engineering Saboteur scenarios (C-01..C-08)",
    )
    p_chaos.add_argument(
        "-s",
        "--scenarios",
        type=str,
        default="all",
        help="Comma-separated scenario IDs or 'all' (default: all)",
    )
    p_chaos.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    p_chaos.add_argument("-q", "--quiet", action="store_true", help="Suppress banner and progress")
    p_chaos.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    p_chaos.set_defaults(handler=_cmd_chaos)

    # 3. kilat train
    p_train = subparsers.add_parser(
        "train",
        help="Run deterministic pre-training from YAML config with preflight Gate 1 & FlightRecorder",
    )
    p_train.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to MainConfig YAML file",
    )
    p_train.add_argument(
        "--set",
        "--override",
        dest="set",
        action="append",
        default=[],
        help="Dot-notation config override (repeatable, e.g. --override training.max_steps=20)",
    )
    p_train.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip fast Gate 1 theoretical preflight verification before training",
    )
    p_train.add_argument("-q", "--quiet", action="store_true", help="Suppress banner")
    p_train.set_defaults(handler=_cmd_train)

    # 4. kilat prepare-data
    p_prep = subparsers.add_parser(
        "prepare-data",
        help="Convert Parquet, JSONL, or text corpus into memmap (.bin/.npy) + .meta.json",
    )
    p_prep.add_argument("-i", "--input", type=str, required=True, help="Input file or Parquet dir")
    p_prep.add_argument("-o", "--output", type=str, required=True, help="Output .bin or .npy path")
    p_prep.add_argument(
        "--tokenizer",
        type=str,
        default="byte",
        help="Tokenizer name/path or 'byte' for offline ByteLevelDebugTokenizer",
    )
    p_prep.add_argument("--key-name", type=str, default="input_ids", help="Token list column/key")
    p_prep.add_argument("--text-key", type=str, default="text", help="Raw text column/key")
    p_prep.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "int32", "int64"],
        help="Target integer dtype for memmap storage (default: uint16)",
    )
    p_prep.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
    p_prep.add_argument("-q", "--quiet", action="store_true", help="Suppress banner and progress")
    p_prep.add_argument("--json", action="store_true", help="Emit metadata JSON to stdout")
    p_prep.set_defaults(handler=_cmd_prepare_data)

    # 5. kilat generate
    p_gen = subparsers.add_parser(
        "generate",
        help="Run autoregressive HybridCache text generation from a Kilat checkpoint",
    )
    p_gen.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint directory")
    p_gen.add_argument("-p", "--prompt", type=str, required=True, help="Input text prompt")
    p_gen.add_argument("--max-new-tokens", type=int, default=32, help="Maximum tokens to generate")
    p_gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p_gen.add_argument("--top-k", type=int, default=50, help="Top-k sampling filter")
    p_gen.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus filter")
    p_gen.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty parameter (1.0 = no penalty)",
    )
    p_gen.add_argument("--do-sample", action="store_true", help="Enable stochastic sampling")
    p_gen.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', 'cuda')")
    p_gen.add_argument("-q", "--quiet", action="store_true", help="Suppress banner")
    p_gen.add_argument("--json", action="store_true", help="Emit JSON output to stdout")
    p_gen.set_defaults(handler=_cmd_generate)

    # 6. kilat export-hf
    p_export = subparsers.add_parser(
        "export-hf",
        help="Export a Kilat checkpoint to Hugging Face / safetensors directory layout",
    )
    p_export.add_argument("-c", "--checkpoint", type=str, required=True, help="Checkpoint directory")
    p_export.add_argument("-o", "--output", type=str, required=True, help="Output HF directory")
    p_export.add_argument("--config", type=str, default=None, help="Optional explicit config YAML")
    p_export.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Write pytorch_model.bin instead of model.safetensors",
    )
    p_export.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p_export.add_argument("--strict", action="store_true", help="Require strict state_dict match")
    p_export.add_argument("--skip-validation", action="store_true", help="Skip forward validation")
    p_export.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer export")
    p_export.add_argument("-q", "--quiet", action="store_true", help="Suppress banner and logs")
    p_export.add_argument("--json", action="store_true", help="Emit JSON summary to stdout")
    p_export.set_defaults(handler=_cmd_export_hf)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for the ``kilat`` CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        sys.stderr.write(f"[kilat {args.command}] ERROR: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
