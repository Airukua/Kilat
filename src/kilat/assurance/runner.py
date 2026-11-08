from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from typing import Any, Callable, Iterable, Optional

import torch

from kilat.models.kernels.flash_decay import HAS_TRITON
from kilat.models.transformer import KilatTransformer
from kilat.assurance.gates import (
    run_gate1_theory,
    run_gate2_numerical,
    run_gate3_equivalence,
    run_gate4_data,
    run_gate5_dynamics,
    run_gate6_system,
)
from kilat.assurance.scaling import (
    run_stage7_scaling_harness,
    run_stage8_readiness_signoff,
)
from kilat.assurance.types import (
    AssuranceError,
    AssuranceReport,
    GateResult,
)
from kilat.configs.main_config import MainConfig
from kilat.configs.model_config import KilatConfig
from kilat.data.collator import KilatDataCollator
from kilat.data.dataloader import build_dataloaders_from_config
from kilat.data.dataset import SyntheticTokenDataset
from torch.utils.data import DataLoader


def _detect_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _compute_config_sha256(cfg_obj: Any) -> str:
    if hasattr(cfg_obj, "to_dict"):
        raw = json.dumps(cfg_obj.to_dict(), sort_keys=True, default=str)
    else:
        raw = str(cfg_obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _execute_gate(
    gate_id: str,
    fn: Callable[[], GateResult],
    keep_going: bool,
    results: list[GateResult],
) -> bool:
    """Execute one gate callable; returns True if passed, False if failed."""
    t0 = time.perf_counter()
    try:
        res = fn()
        results.append(res)
        return res.status == "PASS"
    except AssuranceError as exc:
        dur = (time.perf_counter() - t0) * 1000.0
        results.append(
            GateResult(
                gate_id=gate_id,
                status="FAIL",
                duration_ms=dur,
                metrics=exc.details,
                failure_code=exc.code.value,
                failure_message=exc.message,
            )
        )
        if not keep_going:
            return False
        return True
    except Exception as exc:
        dur = (time.perf_counter() - t0) * 1000.0
        results.append(
            GateResult(
                gate_id=gate_id,
                status="FAIL",
                duration_ms=dur,
                metrics={},
                failure_code="ERR_UNEXPECTED_EXCEPTION",
                failure_message=f"{type(exc).__name__}: {exc}",
            )
        )
        if not keep_going:
            return False
        return True


def run_verification_pipeline(
    config: Optional[MainConfig | KilatConfig] = None,
    output_dir: Optional[str] = None,
    full: bool = False,
    keep_going: bool = False,
    model: Optional[KilatTransformer] = None,
    train_dataloader: Optional[Iterable[Any]] = None,
    eval_dataloader: Optional[Iterable[Any]] = None,
    memmap_path: Optional[str] = None,
    memmap_dtype: str = "uint16",
) -> AssuranceReport:
    """
    Execute Kilat's Layer-1 Pre-Flight Validation Pipeline (`kilat verify`).

    Runs Gates 1..6 sequentially (plus Stage 7..8 when `full=True`) and optionally writes
    `assurance_report.json` and `assurance_report.md` to `output_dir`.
    """
    main_cfg: Optional[MainConfig] = None
    if isinstance(config, MainConfig):
        main_cfg = config
        model_cfg = config.model
    elif isinstance(config, KilatConfig):
        model_cfg = config
    elif model is not None:
        model_cfg = model.config
    else:
        model_cfg = KilatConfig(
            vocab_size=128,
            n_embd=32,
            n_head=4,
            n_layer=2,
            max_position_embeddings=32,
            dropout=0.0,
            use_rope=True,
            rope_head_dim=4,
        )

    target_model = model if model is not None else KilatTransformer(model_cfg)

    if train_dataloader is None:
        if main_cfg is not None:
            train_dl, eval_dl = build_dataloaders_from_config(main_cfg)
            train_dataloader = train_dl
            if eval_dataloader is None:
                eval_dataloader = eval_dl
        else:
            max_pos = int(getattr(model_cfg, "max_position_embeddings", getattr(model_cfg, "block_size", 32)))
            seq_len = min(16, max(8, max_pos))
            collator = KilatDataCollator(pad_token_id=0, max_length=seq_len)
            train_ds = SyntheticTokenDataset(
                vocab_size=model_cfg.vocab_size,
                seq_len=seq_len,
                num_samples=16,
                seed=101,
            )
            eval_ds = SyntheticTokenDataset(
                vocab_size=model_cfg.vocab_size,
                seq_len=seq_len,
                num_samples=8,
                seed=202,
            )
            train_dataloader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collator)
            if eval_dataloader is None:
                eval_dataloader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collator)

    wd = float(main_cfg.training.weight_decay) if main_cfg is not None else 0.1
    resolved_memmap_path: Optional[str] = memmap_path
    resolved_memmap_dtype = memmap_dtype
    if resolved_memmap_path is None and main_cfg is not None and main_cfg.dataloader.dataset_format == "memmap":
        resolved_memmap_path = main_cfg.dataloader.train_data_path
        resolved_memmap_dtype = str(getattr(main_cfg.dataloader, "memmap_dtype", memmap_dtype))

    gate_results: list[GateResult] = []
    pipeline_steps: list[tuple[str, Callable[[], GateResult]]] = [
        ("G1_THEORY_INIT", lambda: run_gate1_theory(model_cfg, model=target_model, weight_decay=wd)),
        ("G2_NUMERICAL_CAUSAL", lambda: run_gate2_numerical(model_cfg, model=target_model)),
        ("G3_EQUIVALENCE", lambda: run_gate3_equivalence(model_cfg, model=target_model)),
        (
            "G4_DATA_FORENSICS",
            lambda: run_gate4_data(
                train_dataloader=train_dataloader,  # type: ignore[arg-type]
                vocab_size=int(model_cfg.vocab_size),
                eval_dataloader=eval_dataloader,
                memmap_path=resolved_memmap_path,
                memmap_dtype=resolved_memmap_dtype,
            ),
        ),
        ("G5_MICRO_DYNAMICS", lambda: run_gate5_dynamics(model_cfg, model=target_model)),
        ("G6_SYSTEM_PARITY", lambda: run_gate6_system(model_cfg)),
    ]

    if full:
        pipeline_steps.append(("S7_SCALING_LAWS", run_stage7_scaling_harness))
        pipeline_steps.append(
            ("S8_READINESS_SIGNOFF", lambda: run_stage8_readiness_signoff(main_cfg, gate_results))
        )

    for gid, step_fn in pipeline_steps:
        should_continue = _execute_gate(gid, step_fn, keep_going=keep_going, results=gate_results)
        if not should_continue and not keep_going:
            break

    overall = "PASS" if all(g.status == "PASS" for g in gate_results) and len(gate_results) > 0 else "FAIL"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    report = AssuranceReport(
        git_commit=_detect_git_commit(),
        config_sha256=_compute_config_sha256(main_cfg if main_cfg is not None else model_cfg),
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device_str,
            "has_triton": bool(HAS_TRITON),
        },
        overall_status=overall,
        gates=gate_results,
    )

    if output_dir is not None:
        report.save(output_dir)

    return report
