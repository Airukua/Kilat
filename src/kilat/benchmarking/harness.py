"""Master orchestrator for Kilat Model Architecture Certification (KMCP)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .pillars.p1_correctness import CorrectnessConfig, run_pillar1
from .pillars.p2_capability import CapabilityConfig, run_pillar2
from .pillars.p3_scaling import ScalingConfig, run_pillar3
from .pillars.p4_efficiency import EfficiencyConfig, run_pillar4
from .pillars.p5_diagnostics import DiagnosticsConfig, run_pillar5
from .registry import build_model, list_models
from .reporter import write_certification_dossier


def load_certification_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_pillar_status(passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"    Status: {status}")


class CertificationHarness:
    """Unified harness executing the 5 certification pillars."""

    def __init__(self, raw_cfg: Optional[Dict[str, Any]] = None):
        self.raw_cfg = raw_cfg or {}
        self.p1_cfg = CorrectnessConfig(**self.raw_cfg.get("pillar1", {}))
        self.p2_cfg = CapabilityConfig(**self.raw_cfg.get("pillar2", {}))
        self.p3_cfg = ScalingConfig(**self.raw_cfg.get("pillar3", {}))
        self.p4_cfg = EfficiencyConfig(**self.raw_cfg.get("pillar4", {}))
        self.p5_cfg = DiagnosticsConfig(**self.raw_cfg.get("pillar5", {}))

    def certify_model(
        self,
        model_name: str,
        output_dir: str = "verify_signoff",
        pillars: Optional[List[str]] = None,
        baseline_name: Optional[str] = "reference-attention",
    ) -> Dict[str, Any]:
        """Run requested pillars on a single registered model."""
        pillars = pillars or ["p1", "p2", "p3", "p4", "p5"]
        print("\n=======================================================")
        print(f"⚡ Starting KMCP Certification for: {model_name}")
        print("=======================================================")

        def builder(**kw):
            return build_model(model_name, **kw)

        def base_builder(**kw):
            return build_model(baseline_name, **kw)

        has_baseline = (
            baseline_name is not None
            and baseline_name in list_models()
            and baseline_name != model_name
        )
        comparative_builder = base_builder if has_baseline else None

        results: Dict[str, Any] = {}

        if "p1" in pillars:
            print("--> [Pillar 1/5] Running Mechanical & Causal Correctness...")
            results["p1"] = run_pillar1(builder, self.p1_cfg)
            _print_pillar_status(bool(results["p1"]["passed"]))

        if "p2" in pillars:
            print(f"--> [Pillar 2/5] Running Mechanistic Capability ({self.p2_cfg.task})...")
            results["p2"] = run_pillar2(builder, self.p2_cfg, arch_name=model_name)
            _print_pillar_status(bool(results["p2"]["passed"]))

        if "p3" in pillars:
            print("--> [Pillar 3/5] Running Empirical Scaling Laws...")
            results["p3"] = run_pillar3(builder, self.p3_cfg)
            _print_pillar_status(bool(results["p3"]["passed"]))

        if "p4" in pillars:
            print("--> [Pillar 4/5] Running Hardware Efficiency Dynamics...")
            results["p4"] = run_pillar4(builder, self.p4_cfg, baseline_builder=comparative_builder)
            _print_pillar_status(bool(results["p4"]["passed"]))

        if "p5" in pillars:
            print("--> [Pillar 5/5] Running Mechanistic Diagnostics...")
            results["p5"] = run_pillar5(builder, self.p5_cfg)
            _print_pillar_status(bool(results["p5"]["passed"]))

        dossier_path = write_certification_dossier(
            output_dir=output_dir,
            model_name=model_name,
            p1=results.get("p1"),
            p2=results.get("p2"),
            p3=results.get("p3"),
            p4=results.get("p4"),
            p5=results.get("p5"),
        )
        print(f"--> Certification Dossier saved to: {dossier_path}")
        return results
