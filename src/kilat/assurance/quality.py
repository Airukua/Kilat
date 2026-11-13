"""
Kilat Code Quality & Static Hygiene Auditor.

Audits Python sources with pure AST analysis (< 0.1s runtime, 0% CPU overhead)
for clean architecture invariants and SonarLint / SonarQube rules:
- python:S107  (Function/method parameter count > 13)
- python:S5713 (Redundant Exception in except clause)
- python:S3457 (Static f-strings without replacement fields)
- python:S1192 (Duplicated string literals >= 4 times)
- python:S8513 (Chained startswith calls)
- python:S3358 (Nested ternary expressions)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class QualityFinding:
    """Represents a static code smell or quality defect detected in AST."""

    rule: str
    location: str
    lineno: int
    detail: str


def audit_python_code(code: str, filename: str = "<string>") -> List[QualityFinding]:
    """
    Parse and analyze Python source code for SonarLint code smells.

    Parameters
    ----------
    code : str
        Source code to analyze.
    filename : str
        Filename for location reporting.

    Returns
    -------
    List[QualityFinding]
        List of detected quality findings ordered by line number.
    """
    findings: List[QualityFinding] = []
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as exc:
        findings.append(QualityFinding("PARSE-ERR", f"{filename}:{exc.lineno}", exc.lineno or 0, str(exc)))
        return findings

    # 1. python:S5713 - Redundant exceptions in except handler
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Tuple):
            classes = [ast.unparse(e) for e in node.type.elts]
            if "Exception" in classes and len(classes) > 1:
                findings.append(
                    QualityFinding(
                        "python:S5713",
                        f"{filename}:{node.lineno}",
                        node.lineno,
                        f"Redundant exception in except ({', '.join(classes)})",
                    )
                )
            elif "BaseException" in classes and len(classes) > 1:
                findings.append(
                    QualityFinding(
                        "python:S5713",
                        f"{filename}:{node.lineno}",
                        node.lineno,
                        f"Redundant exception in except ({', '.join(classes)})",
                    )
                )

    # 2. python:S3457 - Static f-strings without variables (exclude format_spec inside FormattedValue)
    format_specs: Set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            format_specs.add(node.format_spec)

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and node not in format_specs:
            has_formatted = any(isinstance(v, ast.FormattedValue) for v in node.values)
            if (
                not has_formatted
                and len(node.values) == 1
                and isinstance(node.values[0], ast.Constant)
                and isinstance(node.values[0].value, str)
            ):
                val = node.values[0].value
                snippet = val.replace("\n", "\\n")[:40]
                findings.append(
                    QualityFinding(
                        "python:S3457",
                        f"{filename}:{node.lineno}",
                        node.lineno,
                        f"f-string without placeholders: f\"{snippet}\"",
                    )
                )

    # 3. python:S107 - Parameter count > 13 (exclude Triton JIT GPU kernels which require flat pointers)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_triton = any(
                (isinstance(d, ast.Attribute) and d.attr == "jit") or (isinstance(d, ast.Name) and d.id == "jit")
                for d in node.decorator_list
            )
            if is_triton:
                continue

            args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
            count = len(args)
            if count > 13:
                findings.append(
                    QualityFinding(
                        "python:S107",
                        f"{filename}:{node.lineno}",
                        node.lineno,
                        f"Function '{node.name}' has {count} parameters (> 13 authorized)",
                    )
                )

    # 4. python:S1192 - Duplicate string literals (min length 10, count >= 4)
    strings: Dict[str, List[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if len(s) >= 10 and not s.startswith("\n") and not s.endswith("\n"):
                strings.setdefault(s, []).append(node.lineno)
    for s, lines in strings.items():
        if len(lines) >= 4 and not s.startswith("_") and not s.endswith(".py"):
            snippet = s.replace("\n", "\\n")[:30]
            findings.append(
                QualityFinding(
                    "python:S1192",
                    f"{filename}:{lines[0]}",
                    lines[0],
                    f"Literal \"{snippet}\" duplicated {len(lines)} times (lines {lines[:4]})",
                )
            )

    # 5. python:S8513 - Chained startswith
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            calls = [
                v
                for v in node.values
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "startswith"
            ]
            if len(calls) >= 2:
                call_objs = [ast.unparse(c.func.value) for c in calls]  # type: ignore[attr-defined]
                if len(set(call_objs)) < len(call_objs):
                    findings.append(
                        QualityFinding(
                            "python:S8513",
                            f"{filename}:{node.lineno}",
                            node.lineno,
                            "Chained startswith calls can be combined into tuple argument",
                        )
                    )

    # 6. python:S3358 - Nested ternary expressions
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp):
            if isinstance(node.body, ast.IfExp) or isinstance(node.orelse, ast.IfExp):
                findings.append(
                    QualityFinding(
                        "python:S3358",
                        f"{filename}:{node.lineno}",
                        node.lineno,
                        "Nested ternary conditional expression",
                    )
                )

    findings.sort(key=lambda x: x.lineno)
    return findings


def audit_python_file(path: Path | str) -> List[QualityFinding]:
    """Audit a single python source file."""
    path_obj = Path(path)
    if not path_obj.is_file() or not path_obj.name.endswith(".py"):
        return []
    try:
        content = path_obj.read_text(encoding="utf-8")
        return audit_python_code(content, filename=str(path_obj))
    except Exception as exc:
        return [QualityFinding("IO-ERR", f"{path}:0", 0, str(exc))]


def audit_codebase(roots: Sequence[Path | str]) -> Tuple[List[QualityFinding], int]:
    """
    Recursively audit all python files across target directory roots.

    Returns
    -------
    Tuple[List[QualityFinding], int]
        List of all findings and total number of files scanned.
    """
    all_findings: List[QualityFinding] = []
    files_scanned = 0

    for root in roots:
        root_path = Path(root)
        if root_path.is_file():
            findings = audit_python_file(root_path)
            all_findings.extend(findings)
            files_scanned += 1
            continue

        for dirpath, _, filenames in os.walk(root_path):
            if "__pycache__" in dirpath or ".venv" in dirpath or ".git" in dirpath:
                continue
            for file in sorted(filenames):
                if file.endswith(".py"):
                    full_p = Path(dirpath) / file
                    findings = audit_python_file(full_p)
                    all_findings.extend(findings)
                    files_scanned += 1

    return all_findings, files_scanned
