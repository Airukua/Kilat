"""
Behavior-Driven Development (BDD) specification framework for KILAT tests.
Follows the standards defined in UNIT_TESTING_STRUCTURE.md and ml_integration_testing_structure_md.md.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional


class _BDDStep:
    def __init__(self, step_type: str, text: str):
        self.step_type = step_type
        self.text = text
        self.duration_ms: float = 0.0
        self.passed: bool = True
        self.error: Optional[Exception] = None


class _BDDContext:
    def __init__(self):
        self.current_scenario: Optional[str] = None
        self.current_given: Optional[str] = None
        self.current_when: Optional[str] = None
        self.steps: List[_BDDStep] = []
        self.current_file: str = ""

    def reset(self):
        self.current_scenario = None
        self.current_given = None
        self.current_when = None
        self.steps = []


_GLOBAL_BDD_CONTEXT = _BDDContext()


def get_test_classification(file_path: str) -> str:
    norm_path = file_path.replace("\\", "/")
    if "tests/unit" in norm_path:
        return "unit"
    if "tests/benchmarking" in norm_path:
        return "benchmark"
    if "tests/integration" in norm_path:
        return "integration"
    if "tests/component" in norm_path:
        return "component"
    if "tests/contract" in norm_path:
        return "contract"
    if "tests/e2e" in norm_path:
        return "e2e"
    return "unit"


class Scenario:
    def __init__(self, name: str, fn: Optional[Callable[..., Any]] = None):
        self.name = name
        self.fn = fn

    def __enter__(self):
        _GLOBAL_BDD_CONTEXT.reset()
        _GLOBAL_BDD_CONTEXT.current_scenario = self.name
        # Detect caller file path
        frame = inspect.stack()[1]
        _GLOBAL_BDD_CONTEXT.current_file = frame.filename
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Format and flush output according to standard
        _flush_bdd_report(exc_val)
        return False

    def __call__(self, *args, **kwargs):
        with self:
            if self.fn:
                return self.fn(*args, **kwargs)


class Given:
    def __init__(self, condition: str, fn: Optional[Callable[..., Any]] = None):
        self.condition = condition
        self.fn = fn

    def __enter__(self):
        _GLOBAL_BDD_CONTEXT.current_given = self.condition
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, *args, **kwargs):
        with self:
            if self.fn:
                return self.fn(*args, **kwargs)


class When:
    def __init__(self, action: str, fn: Optional[Callable[..., Any]] = None):
        self.action = action
        self.fn = fn

    def __enter__(self):
        _GLOBAL_BDD_CONTEXT.current_when = self.action
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, *args, **kwargs):
        with self:
            if self.fn:
                return self.fn(*args, **kwargs)


class Then:
    def __init__(self, behavior: str, fn: Optional[Callable[..., Any]] = None):
        self.behavior = behavior
        self.fn = fn
        self.step = _BDDStep("Then", behavior)
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        if exc_val is not None:
            self.step.passed = False
            self.step.error = exc_val
        _GLOBAL_BDD_CONTEXT.steps.append(self.step)
        return False

    def __call__(self, *args, **kwargs):
        with self:
            if self.fn:
                return self.fn(*args, **kwargs)


class And:
    def __init__(self, expectation: str, fn: Optional[Callable[..., Any]] = None):
        self.expectation = expectation
        self.fn = fn
        self.step = _BDDStep("And", expectation)
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        if exc_val is not None:
            self.step.passed = False
            self.step.error = exc_val
        _GLOBAL_BDD_CONTEXT.steps.append(self.step)
        return False

    def __call__(self, *args, **kwargs):
        with self:
            if self.fn:
                return self.fn(*args, **kwargs)


def _flush_bdd_report(exception: Optional[BaseException] = None) -> None:
    ctx = _GLOBAL_BDD_CONTEXT
    if not ctx.current_scenario:
        return

    # Colors
    GREEN_BOLD = "\033[1;32m"
    RED_BOLD = "\033[1;31m"
    GRAY = "\033[90m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    is_pass = exception is None and all(s.passed for s in ctx.steps)
    status_tag = f"{GREEN_BOLD}PASS{RESET}" if is_pass else f"{RED_BOLD}FAIL{RESET}"

    # Relative path calculation
    rel_path = os.path.relpath(ctx.current_file, os.getcwd()) if ctx.current_file else "test"
    classification = get_test_classification(rel_path)

    # Clean Scenario Title (remove 'Unit: ' or 'Integration: ' if desired or keep clean)
    scenario_title = ctx.current_scenario
    if scenario_title.startswith("Unit: "):
        scenario_title = scenario_title[6:]
    elif scenario_title.startswith("Integration: "):
        scenario_title = scenario_title[13:]
    elif scenario_title.startswith("Benchmark: "):
        scenario_title = scenario_title[11:]
    elif scenario_title.startswith("Benchmarking: "):
        scenario_title = scenario_title[14:]

    lines = []
    lines.append(f"\n{status_tag}  {CYAN}{classification:<11}{RESET}  {rel_path}")
    lines.append(f"{scenario_title}")

    if ctx.current_given:
        lines.append(f"  given {ctx.current_given}")
    if ctx.current_when:
        lines.append(f"    when {ctx.current_when}")

    indent = "      " if ctx.current_when else ("    " if ctx.current_given else "  ")

    for step in ctx.steps:
        duration_str = f"{step.duration_ms:.1f} ms" if step.duration_ms >= 1.0 else "<1 ms"
        if step.passed:
            lines.append(f"{indent}{GREEN_BOLD}✓{RESET} {step.text} {GRAY}({duration_str}){RESET}")
        else:
            lines.append(f"{indent}{RED_BOLD}✗{RESET} {step.text} {GRAY}({duration_str}){RESET}")

    output_str = "\n".join(lines)
    # Output to stdout directly or stderr so pytest capture can see it or pass through
    sys.stdout.write(output_str + "\n")
    sys.stdout.flush()
