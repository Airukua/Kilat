"""Kilat Model Architecture Certification & Benchmarking Suite (KMCP)."""
from .harness import CertificationHarness, load_certification_config
from .protocol import run_with_seeds, set_seed, summarize
from .registry import build_model, list_models, register_model
from .reporter import write_certification_dossier

__all__ = [
    "CertificationHarness",
    "load_certification_config",
    "register_model",
    "build_model",
    "list_models",
    "set_seed",
    "run_with_seeds",
    "summarize",
    "write_certification_dossier",
]
