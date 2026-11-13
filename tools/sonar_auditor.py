#!/usr/bin/env python3
"""
tools/sonar_auditor.py — Kilat SonarLint & Code Quality Static Hygiene Auditor.

Audits repository files with pure AST analysis (< 0.1s runtime, 0% CPU overhead)
for clean architecture invariants and SonarLint / SonarQube rules:
  - python:S107  (Function/method parameter count > 13)
  - python:S5713 (Redundant Exception in except clause)
  - python:S3457 (Static f-strings without replacement fields)
  - python:S1192 (Duplicated string literals >= 4 times)
  - python:S8513 (Chained startswith calls)
  - python:S3358 (Nested ternary expressions)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path so it works seamlessly from root or subdirectories
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from kilat.assurance.quality import audit_codebase, audit_python_file  # noqa: E402


def main() -> int:
    scan_dirs = ["src", "tests"] if len(sys.argv) <= 1 else sys.argv[1:]
    target_paths = [ROOT_DIR / d if not Path(d).is_absolute() else Path(d) for d in scan_dirs]

    print("=" * 70)
    print("⚡ KILAT SONARLINT & CODE QUALITY HYGIENE AUDITOR")
    print("=" * 70)

    findings, file_count = audit_codebase(target_paths)

    # Group findings by file
    by_file: dict[str, list] = {}
    for f in findings:
        by_file.setdefault(f.location.split(":")[0], []).append(f)

    for file_path, file_findings in sorted(by_file.items()):
        print(f"\n📄 {file_path} ({len(file_findings)} issues):")
        for finding in file_findings:
            print(f"  Line {finding.lineno:4d} [{finding.rule}]: {finding.detail}")

    print("\n" + "=" * 70)
    print(f"Summary: Found {len(findings)} issues across {file_count} files scanned.")
    print("=" * 70)
    return 0 if len(findings) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
