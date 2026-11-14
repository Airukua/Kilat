from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def unwrap_kilat_model(model: Any) -> Any:
    """
    Unwrap PEFT (`PeftModel`) or Hugging Face wrapper layers to reach the underlying
    `KilatTransformer` instance holding `.layers`, `.wte`, and `._last_loss_breakdown`.
    """
    curr = model
    for _ in range(5):
        if hasattr(curr, "layers") and hasattr(curr, "wte"):
            return curr
        if hasattr(curr, "get_base_model") and callable(curr.get_base_model):
            base = curr.get_base_model()
            if base is not None and base is not curr:
                curr = base
                continue
        if hasattr(curr, "base_model") and curr.base_model is not curr:
            curr = curr.base_model
            continue
        if hasattr(curr, "model") and curr.model is not curr:
            curr = curr.model
            continue
        break
    return curr



class AssuranceFailureCode(str, Enum):
    """Structured failure codes for Kilat's 3-Layer Assurance & Chaos Pipeline."""

    # Gate 0: Security & Supply-Chain Posture (OWASP LLM Top 10 / ML Top 10, NCSC Secure AI)
    ERR_INSECURE_DESERIALIZATION = "ERR_INSECURE_DESERIALIZATION"
    ERR_REMOTE_CODE_EXECUTION = "ERR_REMOTE_CODE_EXECUTION"
    ERR_SHELL_INJECTION_RISK = "ERR_SHELL_INJECTION_RISK"
    ERR_UNTRUSTED_DYNAMIC_IMPORT = "ERR_UNTRUSTED_DYNAMIC_IMPORT"
    ERR_CODE_QUALITY_VIOLATION = "ERR_CODE_QUALITY_VIOLATION"

    # Gate 1: Theory & Init
    ERR_UNBIASED_INIT_LOSS = "ERR_UNBIASED_INIT_LOSS"
    ERR_RESIDUAL_VARIANCE_EXPLOSION = "ERR_RESIDUAL_VARIANCE_EXPLOSION"
    ERR_WEIGHT_DECAY_LEAK = "ERR_WEIGHT_DECAY_LEAK"

    # Gate 2: Numerical, Gradient, & Causal Integrity
    ERR_GRADCHECK_MISMATCH = "ERR_GRADCHECK_MISMATCH"
    ERR_CAUSAL_FUTURE_LEAK = "ERR_CAUSAL_FUTURE_LEAK"
    ERR_BATCH_ISOLATION_BREACH = "ERR_BATCH_ISOLATION_BREACH"
    ERR_MISSING_PARAMETER_GRAD = "ERR_MISSING_PARAMETER_GRAD"
    ERR_ZERO_PARAMETER_GRAD = "ERR_ZERO_PARAMETER_GRAD"
    ERR_FROZEN_PARAMETER_GRAD = "ERR_FROZEN_PARAMETER_GRAD"

    # Gate 3: Implementation Equivalence
    ERR_EQUIVALENCE_DRIFT = "ERR_EQUIVALENCE_DRIFT"
    ERR_DECAY_NORMALIZATION_DIVERGENCE = "ERR_DECAY_NORMALIZATION_DIVERGENCE"
    ERR_CACHE_EQUIVALENCE_DRIFT = "ERR_CACHE_EQUIVALENCE_DRIFT"
    ERR_MOE_EQUIVALENCE_DRIFT = "ERR_MOE_EQUIVALENCE_DRIFT"

    # Gate 4: Data Forensics
    ERR_TOKEN_OUT_OF_BOUNDS = "ERR_TOKEN_OUT_OF_BOUNDS"  # noqa: S105 - enum label, not a credential
    ERR_MASK_ALIGNMENT_BREACH = "ERR_MASK_ALIGNMENT_BREACH"
    ERR_EMPTY_TARGET_BATCH = "ERR_EMPTY_TARGET_BATCH"
    ERR_TRAIN_VAL_CONTAMINATION = "ERR_TRAIN_VAL_CONTAMINATION"
    ERR_MEMMAP_INTEGRITY_BREACH = "ERR_MEMMAP_INTEGRITY_BREACH"

    # Gate 5: Micro-Dynamics & Stability
    ERR_OVERFIT_FAILURE = "ERR_OVERFIT_FAILURE"
    ERR_ROUTER_COLLAPSE = "ERR_ROUTER_COLLAPSE"
    ERR_NON_FINITE_ACTIVATION = "ERR_NON_FINITE_ACTIVATION"
    ERR_LR_SCHEDULER_PREMATURE_STEP = "ERR_LR_SCHEDULER_PREMATURE_STEP"

    # Gate 6: System, Memory, Checkpoint & Accumulation/DDP Parity
    ERR_MEMORY_CREEP = "ERR_MEMORY_CREEP"
    ERR_CHECKPOINT_PARITY_DRIFT = "ERR_CHECKPOINT_PARITY_DRIFT"
    ERR_CHECKPOINT_CHECKSUM_MISMATCH = "ERR_CHECKPOINT_CHECKSUM_MISMATCH"
    ERR_ACCUM_PARITY_DRIFT = "ERR_ACCUM_PARITY_DRIFT"

    # Stage 7-8: Scaling & Readiness
    ERR_SCALING_COORDINATE_DRIFT = "ERR_SCALING_COORDINATE_DRIFT"
    ERR_SCALING_MONOTONICITY_BREACH = "ERR_SCALING_MONOTONICITY_BREACH"
    ERR_READINESS_SIGNOFF_FAILED = "ERR_READINESS_SIGNOFF_FAILED"


class AssuranceError(RuntimeError):
    """Exception raised when a pre-flight validation gate detects a mathematical or system invariant breach."""

    def __init__(
        self,
        code: AssuranceFailureCode,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code.value}] {message}")


@dataclass
class GateResult:
    """Structured outcome of a single assurance gate or scaling stage."""

    gate_id: str
    status: str  # "PASS" | "FAIL"
    duration_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gate_id": self.gate_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "metrics": self.metrics,
        }
        if self.failure_code is not None:
            payload["failure_code"] = self.failure_code
        if self.failure_message is not None:
            payload["failure_message"] = self.failure_message
        return payload


@dataclass
class AssuranceReport:
    """Signed pre-training readiness certificate emitted by `kilat verify`."""

    schema_version: str = "1.0"
    kilat_version: str = "0.1.0"
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    git_commit: str = "unknown"
    config_sha256: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    overall_status: str = "PASS"
    gates: list[GateResult] = field(default_factory=list)
    chaos_verification: Optional[dict[str, Any]] = None

    @property
    def passed(self) -> bool:
        return self.overall_status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kilat_version": self.kilat_version,
            "timestamp_utc": self.timestamp_utc,
            "git_commit": self.git_commit,
            "config_sha256": self.config_sha256,
            "environment": self.environment,
            "overall_status": self.overall_status,
            "gates": [g.to_dict() for g in self.gates],
        }
        if self.chaos_verification is not None:
            data["chaos_verification"] = self.chaos_verification
        return data

    def to_markdown(self) -> str:
        lines = [
            "# Kilat Pre-Training Assurance Certificate",
            "",
            f"- **Overall Status**: `{self.overall_status}`",
            f"- **Timestamp (UTC)**: `{self.timestamp_utc}`",
            f"- **Git Commit**: `{self.git_commit}`",
            f"- **Config SHA-256**: `{self.config_sha256}`",
            f"- **Environment**: Python `{self.environment.get('python', 'n/a')}`, "
            f"PyTorch `{self.environment.get('torch', 'n/a')}`, "
            f"Device `{self.environment.get('device', 'n/a')}`, "
            f"Triton `{self.environment.get('has_triton', False)}`",
            "",
            "## Verification Gates",
            "",
            "| Gate ID | Status | Duration (ms) | Failure Code | Key Metrics |",
            "| :--- | :---: | ---: | :--- | :--- |",
        ]
        for g in self.gates:
            compact_metrics = ", ".join(
                f"`{k}={v:.4g}`" if isinstance(v, float) else f"`{k}={v}`"
                for k, v in list(g.metrics.items())[:4]
            )
            lines.append(
                f"| `{g.gate_id}` | **{g.status}** | {g.duration_ms:.2f} | "
                f"{g.failure_code or '-'} | {compact_metrics} |"
            )
        if self.chaos_verification is not None:
            lines.extend(
                [
                    "",
                    "## Active Chaos Verification",
                    "",
                    f"- **Scenarios Executed**: {self.chaos_verification.get('scenarios_total', 0)}",
                    f"- **Scenarios Caught**: {self.chaos_verification.get('scenarios_caught', 0)}",
                    f"- **False Negatives**: {self.chaos_verification.get('false_negatives', 0)}",
                ]
            )
        lines.append("")
        return "\n".join(lines)

    def save(self, output_dir: str) -> tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "assurance_report.json")
        md_path = os.path.join(output_dir, "assurance_report.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        return json_path, md_path
