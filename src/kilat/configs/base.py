from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and return a dictionary.

    WHY: Provides a consistent, safe interface for loading YAML configuration
    files across the codebase. Handles empty files gracefully and uses
    yaml.safe_load() to prevent arbitrary code execution (critical for
    loading untrusted configuration files).

    Assumptions:
        - The YAML file uses UTF-8 encoding (standard for config files).
        - Empty files or files containing only comments return an empty dict.
        - The YAML content should represent a mapping (dictionary); lists at
          top-level will be returned as-is but config system expects dict.

    Edge Cases:
        - If the file is empty or contains only whitespace/comments,
          yaml.safe_load() returns None. We convert None to {} to avoid
          downstream AttributeError when accessing keys.
        - Malformed YAML raises yaml.YAMLError with line number information.
        - File not found raises FileNotFoundError (caller should handle).

    Performance:
        - O(n) where n = file size. Typically < 1ms for config files.
        - No caching; caller should cache if loading same file repeatedly.

    Example:
        >>> config = load_yaml_file("configs/model.yaml")
        >>> vocab_size = config.get("vocab_size", 32000)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # YAML returns None for empty files; convert to empty dict for consistent API
    return data or {}


def dump_yaml_file(data: dict[str, Any], path: str | Path, *, width: int = 120) -> str:
    """
    Dump a dictionary to YAML and write it to disk.

    WHY: Centralises YAML serialisation with project‑wide consistent formatting.
    This ensures all config files have the same style (block format, sorted keys
    preserved, unicode support), making them easier to diff and review in version
    control.

    Design Decisions:
        - default_flow_style=False: Use block style (each key-value on its own
          line) rather than inline JSON-like format. This is more readable for
          config files and plays nicely with git diffs.
        - sort_keys=False: Preserve the natural order of keys as defined in the
          dict or dataclass. This allows logical grouping (e.g., all model params
          together) rather than alphabetical scattering.
        - allow_unicode=True: Preserve non-ASCII characters (e.g., in descriptions)
          instead of escaping them as \\uXXXX.
        - width=120: Allow lines up to 120 characters before wrapping. Wider than
          default (80) is acceptable for config files because they rarely contain
          long lines; when they do, wrapping breaks readability.

    Assumptions:
        - The dictionary contains only YAML-serialisable types (str, int, float,
          bool, list, dict, None). Custom objects must be converted to dict first.
        - The parent directory of `path` may not exist; we create it automatically.

    Edge Cases:
        - If `path` already exists, it is overwritten without warning. This is
          intentional for config export; caller should check existence if needed.
        - If `data` is empty dict, writes a file containing "{}" (empty YAML map).

    Parameters:
        data: Dictionary to serialise.
        path: Destination file path. Parent directories are created if missing.
        width: Maximum line width for YAML output (default: 120).

    Returns:
        The YAML string that was written (useful for debugging or chaining).

    Example:
        >>> config = {"model": {"n_embd": 768, "n_layer": 12}}
        >>> dump_yaml_file(config, "configs/exported.yaml")
        >>> # File written with block-style YAML
    """
    yaml_str = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=width,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml_str)
    return yaml_str