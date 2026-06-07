from __future__ import annotations

import inspect
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _safe_import():
    try:
        from utils.config import KilatConfig
        from arc.model import KilatTransformer
        return KilatConfig, KilatTransformer, None
    except Exception as exc:
        return None, None, exc


def _show_class_info(kilat_config_cls) -> None:
    _print_header("Import Info")
    config_file = Path(inspect.getfile(kilat_config_cls)).resolve()
    print(f"KilatConfig file: {config_file}")
    print(f"Repo root:        {REPO_ROOT}")
    print(f"Inside repo?:      {REPO_ROOT in config_file.parents}")
    print(f"sys.path[0]:       {sys.path[0] if sys.path else '<empty>'}")
    print(f"Python:            {sys.version.split()[0]}")

    _print_header("Class Signature")
    print(f"__init__   : {inspect.signature(kilat_config_cls.__init__)}")
    print(f"__post_init__: {inspect.signature(kilat_config_cls.__post_init__)}")
    print(f"MRO        : {[cls.__name__ for cls in kilat_config_cls.mro()]}")


def _show_instance_state(config) -> None:
    _print_header("Instance State")
    keys = [
        "vocab_size",
        "n_embd",
        "n_layer",
        "n_head",
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "use_cache",
        "initializer_range",
        "_output_attentions",
        "_output_hidden_states",
        "_use_cache",
        "_attn_implementation",
        "_attn_implementation_internal",
        "_experts_implementation_internal",
    ]
    for key in keys:
        print(f"{key:32s} -> {getattr(config, key, '<missing>')}")


def _try_instantiate(KilatConfig) -> None:
    _print_header("Instantiation Test")
    try:
        config = KilatConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=4,
            ffn_mode="dense",
        )
        print("KilatConfig(...) succeeded")
        _show_instance_state(config)
    except Exception as exc:
        print(f"KilatConfig(...) failed: {type(exc).__name__}: {exc}")
        print("Traceback:")
        traceback.print_exc()


def _try_model(KilatConfig, KilatTransformer) -> None:
    if KilatTransformer is None:
        return

    _print_header("Model Init Test")
    try:
        config = KilatConfig(
            vocab_size=100,
            n_embd=32,
            n_layer=2,
            n_head=4,
            ffn_mode="dense",
        )
        model = KilatTransformer(config)
        print("KilatTransformer(config) succeeded")
        print(f"Model class file: {Path(inspect.getfile(KilatTransformer)).resolve()}")
        print(f"Config class file: {Path(inspect.getfile(KilatConfig)).resolve()}")
        print(f"Model device param count: {sum(p.numel() for p in model.parameters())}")
    except Exception as exc:
        print(f"KilatTransformer(config) failed: {type(exc).__name__}: {exc}")
        print("Traceback:")
        traceback.print_exc()


def main() -> int:
    KilatConfig, KilatTransformer, import_error = _safe_import()
    if import_error is not None:
        _print_header("Import Error")
        print(f"{type(import_error).__name__}: {import_error}")
        print("Traceback:")
        traceback.print_exc()
        return 1

    _show_class_info(KilatConfig)
    _try_instantiate(KilatConfig)
    _try_model(KilatConfig, KilatTransformer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
