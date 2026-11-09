#!/usr/bin/env python3
"""Unified CLI Entrypoint for Kilat Model Architecture Certification Pipeline (KMCP).

Usage examples:
    python tools/certify.py list-models
    python tools/certify.py run-all --models kilat-v2 reference-attention
    python tools/certify.py p1-correctness --models kilat-v2
    python tools/certify.py p2-capability --models kilat-v2
    python tools/certify.py p4-efficiency --models kilat-v2 --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure src is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from kilat.benchmarking.harness import (  # noqa: E402
    CertificationHarness,
    load_certification_config,
)
from kilat.benchmarking.registry import list_models  # noqa: E402


def parse_args(argv: List[str] = None):
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        type=str,
        default="configs/certification/smoke_test.yaml",
        help="Path to certification YAML configuration file.",
    )
    common_parser.add_argument(
        "--output",
        type=str,
        default="reports/certification",
        help="Directory to save markdown and JSON dossiers.",
    )
    common_parser.add_argument(
        "--models",
        nargs="+",
        default=["kilat-v2"],
        help="One or more registered model names to evaluate.",
    )
    common_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override execution device (e.g. 'cuda' or 'cpu').",
    )

    parser = argparse.ArgumentParser(
        description="⚡ Kilat Model Architecture Certification & Benchmarking Pipeline (KMCP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-models", help="List all registered models in registry.", parents=[common_parser])
    subparsers.add_parser("run-all", help="Execute all 5 certification pillars.", parents=[common_parser])
    subparsers.add_parser("p1-correctness", help="Execute Pillar 1: Mechanical & Causal Correctness.", parents=[common_parser])
    subparsers.add_parser("p2-capability", help="Execute Pillar 2: Synthetic Reasoning & Recall Capacity.", parents=[common_parser])
    subparsers.add_parser("p3-scaling", help="Execute Pillar 3: Empirical Scaling Laws & Quality.", parents=[common_parser])
    subparsers.add_parser("p4-efficiency", help="Execute Pillar 4: Hardware & Throughput Dynamics.", parents=[common_parser])
    subparsers.add_parser("p5-diagnostics", help="Execute Pillar 5: Mechanistic Diagnostics & Forensics.", parents=[common_parser])

    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.command == "list-models":
        print(f"Registered Models: {list_models()}")
        return 0

    cfg_dict = {}
    if Path(args.config).is_file():
        cfg_dict = load_certification_config(args.config)

    # Apply device override if requested
    if args.device:
        for p in ["pillar1", "pillar2", "pillar3", "pillar4", "pillar5"]:
            if p not in cfg_dict:
                cfg_dict[p] = {}
            cfg_dict[p]["device"] = args.device

    harness = CertificationHarness(cfg_dict)

    pillar_map = {
        "run-all": ["p1", "p2", "p3", "p4", "p5"],
        "p1-correctness": ["p1"],
        "p2-capability": ["p2"],
        "p3-scaling": ["p3"],
        "p4-efficiency": ["p4"],
        "p5-diagnostics": ["p5"],
    }
    target_pillars = pillar_map[args.command]

    for model_name in args.models:
        harness.certify_model(
            model_name=model_name,
            output_dir=args.output,
            pillars=target_pillars,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
