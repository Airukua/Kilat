"""
Layer-0 security primitives for Kilat (stdlib only, zero intra-package imports).

These guards close the supply-chain and deserialization attack surface of an LLM
training/inference toolkit, mapped to:

- OWASP Top 10 for LLM Applications (2025): LLM03 Supply Chain, LLM04 Data & Model Poisoning.
- OWASP Machine Learning Security Top 10: ML06 AI Supply Chain Attacks, ML10 Model Poisoning.
- OWASP Top 10 (2021): A01 Broken Access Control (path traversal), A08 Software & Data Integrity Failures.
- NCSC / CISA "Guidelines for Secure AI System Development": secure development (supply chain,
  asset protection) and secure deployment (protect models from tampering).

Threat model: a checkpoint directory, Hub snapshot, tokenizer config, or shard index is
*untrusted input*. Loading it must never execute attacker-chosen code or read files outside
the checkpoint root.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Union

#: Comma-separated extra module prefixes the operator explicitly trusts for dynamic import.
TRUSTED_MODULES_ENV = "KILAT_TRUSTED_MODULES"
#: Operator-provided tokenizer builder ("module:function"); its module is implicitly trusted.
TOKENIZER_BUILDER_ENV = "KILAT_TOKENIZER_BUILDER"
#: When set to "1", unpinned Hugging Face Hub revisions are rejected instead of warned about.
REQUIRE_PINNED_REVISION_ENV = "KILAT_REQUIRE_PINNED_REVISION"

#: Module prefixes always trusted for dynamic import (first-party code only).
DEFAULT_TRUSTED_MODULE_PREFIXES: tuple[str, ...] = ("kilat",)

_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class UntrustedModuleError(ImportError):
    """Raised when a config asks Kilat to import a module outside the trusted allowlist.

    Subclasses ``ImportError`` for backward compatibility with callers that already handle
    import failures, but loaders must re-raise it instead of silently falling back.
    """


class UnsafePathError(ValueError):
    """Raised when an untrusted relative path would escape its root directory."""


class UnpinnedRevisionError(ValueError):
    """Raised when a remote artifact is fetched without an immutable commit SHA revision."""


def _split_prefixes(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def trusted_module_prefixes() -> tuple[str, ...]:
    """Return the effective allowlist of importable module prefixes.

    Sources (all operator-controlled, never read from checkpoint/config files):
    1. ``DEFAULT_TRUSTED_MODULE_PREFIXES`` (first-party ``kilat``).
    2. ``KILAT_TRUSTED_MODULES`` environment variable (comma-separated prefixes).
    3. The module part of ``KILAT_TOKENIZER_BUILDER`` if set.
    """
    prefixes = list(DEFAULT_TRUSTED_MODULE_PREFIXES)
    prefixes.extend(_split_prefixes(os.environ.get(TRUSTED_MODULES_ENV, "")))
    env_builder = os.environ.get(TOKENIZER_BUILDER_ENV, "").strip()
    if env_builder:
        prefixes.append(env_builder.rsplit(":", 1)[0])
    return tuple(dict.fromkeys(prefixes))


def is_trusted_module(module_path: str) -> bool:
    """True if ``module_path`` equals or is a submodule of a trusted prefix."""
    if not module_path or module_path.startswith("."):
        return False
    return any(module_path == p or module_path.startswith(p + ".") for p in trusted_module_prefixes())


def assert_trusted_module(module_path: str) -> None:
    """Raise ``UntrustedModuleError`` unless ``module_path`` is on the allowlist."""
    if not is_trusted_module(module_path):
        raise UntrustedModuleError(
            f"Refusing to import untrusted module '{module_path}'. Dynamic imports requested by "
            f"configuration are restricted to {list(trusted_module_prefixes())}. If you wrote this "
            f"module yourself, allow it explicitly: export {TRUSTED_MODULES_ENV}={module_path}"
        )


def safe_join(root: Union[str, Path], untrusted_relative: str) -> Path:
    """Join an untrusted relative path onto ``root``, rejecting traversal and absolute paths.

    Used for shard filenames read from ``*.index.json`` weight maps, which an attacker controls.
    """
    if not isinstance(untrusted_relative, str) or not untrusted_relative:
        raise UnsafePathError(f"Invalid path component: {untrusted_relative!r}")
    if "\x00" in untrusted_relative:
        raise UnsafePathError("Path component contains a NUL byte.")
    candidate = Path(untrusted_relative)
    if candidate.is_absolute() or candidate.drive:
        raise UnsafePathError(f"Absolute path not allowed: {untrusted_relative!r}")
    root_resolved = Path(root).resolve()
    resolved = (root_resolved / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise UnsafePathError(
            f"Path {untrusted_relative!r} escapes root directory {str(root_resolved)!r}."
        ) from None
    return resolved


def is_pinned_revision(revision: object) -> bool:
    """True only for an immutable 40-hex-character commit SHA (branches/tags are mutable)."""
    return isinstance(revision, str) and bool(_COMMIT_SHA_RE.fullmatch(revision))


def pinned_revision_required() -> bool:
    """True if the operator demands pinned revisions for every remote download."""
    return os.environ.get(REQUIRE_PINNED_REVISION_ENV, "").strip() == "1"


__all__ = [
    "DEFAULT_TRUSTED_MODULE_PREFIXES",
    "REQUIRE_PINNED_REVISION_ENV",
    "TOKENIZER_BUILDER_ENV",
    "TRUSTED_MODULES_ENV",
    "UnpinnedRevisionError",
    "UnsafePathError",
    "UntrustedModuleError",
    "assert_trusted_module",
    "is_pinned_revision",
    "is_trusted_module",
    "pinned_revision_required",
    "safe_join",
    "trusted_module_prefixes",
]
