"""
Gate 0 — Security & Supply-Chain Posture (`G0_SECURITY_SUPPLY_CHAIN`).

Runs before any numerical gate. A model that is mathematically perfect but loads attacker
code from a checkpoint is not ready for training. Two checks, both pure-CPU and fast:

1. **Static source scan** (AST, no regex) of the Kilat package for code-execution sinks:

   | Rule        | Pattern                                                          | Code                              | Mapping |
   |-------------|------------------------------------------------------------------|-----------------------------------|---------|
   | SEC-DESER-1 | ``torch.load`` without literal ``weights_only=True``             | ``ERR_INSECURE_DESERIALIZATION``  | OWASP LLM03, ML06; A08 |
   | SEC-DESER-2 | ``pickle``/``cloudpickle``/``dill``/``joblib`` ``load(s)``       | ``ERR_INSECURE_DESERIALIZATION``  | OWASP LLM03, ML06; A08 |
   | SEC-DESER-3 | ``yaml.load``/``yaml.unsafe_load``/``yaml.full_load``            | ``ERR_INSECURE_DESERIALIZATION``  | OWASP A08 |
   | SEC-DESER-4 | ``numpy.load(..., allow_pickle=True)``                           | ``ERR_INSECURE_DESERIALIZATION``  | OWASP A08 |
   | SEC-RCE-1   | ``trust_remote_code=True``                                       | ``ERR_REMOTE_CODE_EXECUTION``     | OWASP LLM03, ML06 |
   | SEC-RCE-2   | builtin ``eval(...)`` / ``exec(...)``                            | ``ERR_REMOTE_CODE_EXECUTION``     | OWASP A03 |
   | SEC-SHELL-1 | ``shell=True``, ``os.system``, ``os.popen``                      | ``ERR_SHELL_INJECTION_RISK``      | OWASP A03 |

2. **Configuration audit**: tokenizer ``custom_builder`` / ``custom_module`` must resolve to a
   trusted module prefix (``kilat`` or ``KILAT_TRUSTED_MODULES``) — ``ERR_UNTRUSTED_DYNAMIC_IMPORT``.

Aligned with NCSC/CISA "Guidelines for Secure AI System Development" (secure development:
identify, track and protect assets; secure your supply chain).
"""

from __future__ import annotations

import ast
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from kilat.assurance.types import AssuranceError, AssuranceFailureCode, GateResult
from kilat.utils.security import is_trusted_module, trusted_module_prefixes

GATE_ID = "G0_SECURITY_SUPPLY_CHAIN"

_PICKLE_MODULES = frozenset({"pickle", "cPickle", "cloudpickle", "dill", "joblib", "_pickle"})
_YAML_UNSAFE = frozenset({"load", "unsafe_load", "full_load", "load_all", "unsafe_load_all", "full_load_all"})

# Severity order: the highest-priority code present becomes the gate failure code.
_CODE_PRIORITY: tuple[AssuranceFailureCode, ...] = (
    AssuranceFailureCode.ERR_REMOTE_CODE_EXECUTION,
    AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
    AssuranceFailureCode.ERR_SHELL_INJECTION_RISK,
    AssuranceFailureCode.ERR_UNTRUSTED_DYNAMIC_IMPORT,
)


@dataclass(frozen=True)
class SecurityFinding:
    rule: str
    code: str
    location: str
    detail: str
    owasp: str


def _kw(call: ast.Call, name: str) -> Optional[ast.expr]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_const_true(node: Optional[ast.expr]) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _dotted(node: ast.expr) -> str:
    """Return 'a.b.c' for Name/Attribute chains, else ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _scan_tree(tree: ast.AST, location_prefix: str) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []

    def add(node: ast.AST, rule: str, code: AssuranceFailureCode, detail: str, owasp: str) -> None:
        findings.append(
            SecurityFinding(
                rule=rule,
                code=code.value,
                location=f"{location_prefix}:{getattr(node, 'lineno', 0)}",
                detail=detail,
                owasp=owasp,
            )
        )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        head, _, tail = name.rpartition(".")

        if name == "torch.load" and not _is_const_true(_kw(node, "weights_only")):
            add(node, "SEC-DESER-1", AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
                "torch.load without weights_only=True unpickles arbitrary objects", "LLM03/ML06/A08")
        elif head in _PICKLE_MODULES and tail in {"load", "loads", "Unpickler"}:
            add(node, "SEC-DESER-2", AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
                f"{name} deserializes arbitrary objects", "LLM03/ML06/A08")
        elif head == "yaml" and tail in _YAML_UNSAFE:
            add(node, "SEC-DESER-3", AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
                f"{name} can construct arbitrary Python objects; use yaml.safe_load", "A08")
        elif name in {"np.load", "numpy.load"} and _is_const_true(_kw(node, "allow_pickle")):
            add(node, "SEC-DESER-4", AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
                "numpy.load(allow_pickle=True) unpickles arbitrary objects", "A08")

        if _is_const_true(_kw(node, "trust_remote_code")):
            add(node, "SEC-RCE-1", AssuranceFailureCode.ERR_REMOTE_CODE_EXECUTION,
                f"{name or '<call>'}(trust_remote_code=True) executes code shipped with the artifact",
                "LLM03/ML06")
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            add(node, "SEC-RCE-2", AssuranceFailureCode.ERR_REMOTE_CODE_EXECUTION,
                f"builtin {node.func.id}() executes dynamic code", "A03")
        if _is_const_true(_kw(node, "shell")) or name in {"os.system", "os.popen"}:
            add(node, "SEC-SHELL-1", AssuranceFailureCode.ERR_SHELL_INJECTION_RISK,
                f"{name or '<call>'} spawns a shell", "A03")
    return findings


def scan_python_sources(root: Path) -> tuple[list[SecurityFinding], int]:
    """AST-scan every ``*.py`` under ``root``. Returns (findings, files_scanned)."""
    findings: list[SecurityFinding] = []
    files = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files += 1
        rel = path.relative_to(root.parent).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(SecurityFinding("SEC-PARSE", "ERR_UNEXPECTED_EXCEPTION", rel, str(exc), "-"))
            continue
        findings.extend(_scan_tree(tree, rel))
    return findings, files


def audit_tokenizer_config(tokenizer_cfg: Any) -> list[SecurityFinding]:
    """Flag tokenizer dynamic-import targets that fall outside the trusted prefixes."""
    findings: list[SecurityFinding] = []
    if tokenizer_cfg is None:
        return findings
    targets: list[tuple[str, str]] = []
    builder = getattr(tokenizer_cfg, "custom_builder", None)
    if builder:
        targets.append(("custom_builder", str(builder).rsplit(":", 1)[0]))
    module = getattr(tokenizer_cfg, "custom_module", None)
    if module:
        targets.append(("custom_module", str(module)))
    for field_name, mod in targets:
        if not is_trusted_module(mod):
            findings.append(
                SecurityFinding(
                    rule="SEC-IMPORT-1",
                    code=AssuranceFailureCode.ERR_UNTRUSTED_DYNAMIC_IMPORT.value,
                    location=f"config.tokenizer.{field_name}",
                    detail=f"module '{mod}' not in trusted prefixes {list(trusted_module_prefixes())}",
                    owasp="LLM03/ML06",
                )
            )
    return findings


def _default_source_root() -> Path:
    # .../src/kilat/assurance/gates/gate0_security.py -> .../src/kilat
    return Path(__file__).resolve().parents[2]


def run_gate0_security(
    main_cfg: Optional[Any] = None,
    source_roots: Optional[Iterable[Path]] = None,
) -> GateResult:
    """Execute Gate 0. Raises ``AssuranceError`` on any finding."""
    t0 = time.perf_counter()
    roots = [Path(r) for r in source_roots] if source_roots is not None else [_default_source_root()]

    findings: list[SecurityFinding] = []
    files_scanned = 0
    for root in roots:
        root_findings, n = scan_python_sources(root)
        findings.extend(root_findings)
        files_scanned += n
    findings.extend(audit_tokenizer_config(getattr(main_cfg, "tokenizer", None)))

    metrics: dict[str, Any] = {
        "files_scanned": files_scanned,
        "findings_count": len(findings),
        "findings": [asdict(f) for f in findings],
        "trusted_module_prefixes": list(trusted_module_prefixes()),
        "frameworks": [
            "OWASP Top 10 for LLM Applications 2025",
            "OWASP Machine Learning Security Top 10",
            "OWASP Top 10 2021",
            "NCSC/CISA Guidelines for Secure AI System Development",
        ],
    }

    if files_scanned == 0:
        raise AssuranceError(
            AssuranceFailureCode.ERR_INSECURE_DESERIALIZATION,
            "Security scan inspected 0 files: refusing to report a vacuous PASS.",
            metrics,
        )

    if findings:
        present = {f.code for f in findings}
        code = next((c for c in _CODE_PRIORITY if c.value in present), _CODE_PRIORITY[0])
        first = next(f for f in findings if f.code == code.value)
        raise AssuranceError(
            code,
            f"{len(findings)} security finding(s); first: [{first.rule}] {first.location} — {first.detail}",
            metrics,
        )

    return GateResult(
        gate_id=GATE_ID,
        status="PASS",
        duration_ms=(time.perf_counter() - t0) * 1000.0,
        metrics=metrics,
    )


__all__ = ["GATE_ID", "SecurityFinding", "audit_tokenizer_config", "run_gate0_security", "scan_python_sources"]
