from __future__ import annotations
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch
import torch.distributed as dist
from .callbacks import CallbackHandler, EarlyStoppingCallback, TrainerState
from .distributed import (
    barrier,
    get_global_rank,
    get_world_size,
    is_distributed_initialized,
)
from .optimizer import compute_total_steps

CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device & distributed helpers
# ---------------------------------------------------------------------------

def get_device(model: torch.nn.Module) -> torch.device:
    """
    Return the device on which the model's parameters reside.

    WHY: After moving a model to a device (e.g., `model.to(device)`), there is
    no direct `model.device` attribute. This helper inspects the first parameter
    to determine the current device. It is useful for creating tensors on the
    same device as the model without carrying around a separate `device` variable.

    Assumptions:
    - The model has at least one parameter (non‑empty model). If not, we fall back
      to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
    - All parameters are on the same device – typical in practice.

    Edge Cases:
    - If the model is a `nn.DataParallel` or `DistributedDataParallel` wrapper,
      the underlying module (`model.module`) is checked.
    - For meta‑device (torch.empty(0)), fallback is used.

    Performance: O(1) — just inspects the first parameter.

    Example Usage
    -------------
        >>> model = nn.Linear(10, 10).to("cuda")
        >>> device = get_device(model)  # device(type='cuda')
    """
    # Unwrap DataParallel / DDP to access the real module
    model = model.module if hasattr(model, "module") else model
    try:
        # Get the first parameter's device
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters – fallback to CUDA if available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def should_save_on_rank0(state: TrainerState) -> bool:
    """
    Return True if the current process should save checkpoints/logs.
    """
    return state.is_world_process_zero


_KEY_TORCH_CUDA = "torch_cuda"
_CHECKPOINT_PREFIX = "checkpoint-"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_KEY_TRAINER_STATE = "trainer_state"
_KEY_CALLBACK_STATES = "callback_states"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """
    Capture full RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA) in a
    pure-primitive + Tensor representation compatible with ``torch.load(weights_only=True)``.
    """
    np_st = np.random.get_state()
    rng_dict: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": (
            str(np_st[0]),
            np_st[1].tolist(),
            int(np_st[2]),
            int(np_st[3]),
            float(np_st[4]),
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        _KEY_TORCH_CUDA: (
            [s.clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }
    return rng_dict


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore full RNG states captured by :func:`capture_rng_state`."""
    if not rng_state:
        return
    if "python" in rng_state and rng_state["python"] is not None:
        py_st = rng_state["python"]
        # Ensure inner MT state is a tuple (JSON or older serialization safety)
        if isinstance(py_st, (list, tuple)) and len(py_st) == 3:
            random.setstate((int(py_st[0]), tuple(int(x) for x in py_st[1]), py_st[2]))
    if "numpy" in rng_state and rng_state["numpy"] is not None:
        np_st = rng_state["numpy"]
        np_keys = np.array(np_st[1], dtype=np.uint32)
        np.random.set_state((str(np_st[0]), np_keys, int(np_st[2]), int(np_st[3]), float(np_st[4])))
    if "torch_cpu" in rng_state and rng_state["torch_cpu"] is not None:
        cpu_st = rng_state["torch_cpu"]
        if isinstance(cpu_st, torch.Tensor):
            torch.set_rng_state(cpu_st.detach().cpu().to(dtype=torch.uint8))
    if (
        torch.cuda.is_available()
        and _KEY_TORCH_CUDA in rng_state
        and rng_state[_KEY_TORCH_CUDA] is not None
    ):
        cuda_states = [
            s.detach().cpu().to(dtype=torch.uint8)
            for s in rng_state[_KEY_TORCH_CUDA][: torch.cuda.device_count()]
            if isinstance(s, torch.Tensor)
        ]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


def compute_file_sha256(file_path: str | Path) -> tuple[str, int]:
    """Compute streaming SHA-256 hex digest and byte size of a file."""
    h = hashlib.sha256()
    total_bytes = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            total_bytes += len(chunk)
    return h.hexdigest(), total_bytes


def write_checkpoint_manifest(
    checkpoint_dir: str | Path,
    step: Optional[int] = None,
    tag: Optional[str] = None,
) -> dict[str, Any]:
    """Write ``checkpoint_manifest.json`` recording SHA-256 and byte sizes of all checkpoint artifacts."""
    ckpt_root = Path(checkpoint_dir)
    files_meta: dict[str, dict[str, Any]] = {}
    for item in sorted(ckpt_root.rglob("*")):
        if item.is_file() and item.name != CHECKPOINT_MANIFEST_NAME:
            rel = item.relative_to(ckpt_root).as_posix()
            sha, size_b = compute_file_sha256(item)
            files_meta[rel] = {"sha256": sha, "size_bytes": size_b}

    manifest = {
        "schema_version": "1.0",
        "step": step,
        "tag": tag,
        "files": files_meta,
    }
    with open(ckpt_root / CHECKPOINT_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_checkpoint_manifest(checkpoint_dir: str | Path) -> None:
    """
    Verify every file recorded in ``checkpoint_manifest.json`` against disk.
    Raises ``RuntimeError`` prefixed with ``ERR_CHECKPOINT_CHECKSUM_MISMATCH``
    upon any missing, truncated, or corrupted artifact.
    """
    ckpt_root = Path(checkpoint_dir)
    manifest_path = ckpt_root / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.exists():
        return  # Backward-compatible with legacy checkpoints lacking a manifest

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"ERR_CHECKPOINT_CHECKSUM_MISMATCH: Corrupted manifest JSON at {manifest_path}: {exc}"
        ) from exc

    files_meta = manifest.get("files", {})
    if not isinstance(files_meta, dict):
        raise RuntimeError(
            f"ERR_CHECKPOINT_CHECKSUM_MISMATCH: Invalid 'files' section in {manifest_path}"
        )

    for rel_path, expected in files_meta.items():
        target = ckpt_root / rel_path
        if not target.is_file():
            raise RuntimeError(
                f"ERR_CHECKPOINT_CHECKSUM_MISMATCH: Missing checkpoint file '{rel_path}' in {ckpt_root}"
            )
        actual_sha, actual_size = compute_file_sha256(target)
        expected_size = expected.get("size_bytes")
        expected_sha = expected.get("sha256")
        if expected_size is not None and actual_size != int(expected_size):
            raise RuntimeError(
                f"ERR_CHECKPOINT_CHECKSUM_MISMATCH: File size mismatch for '{rel_path}' "
                f"(expected {expected_size}B, got {actual_size}B)"
            )
        if expected_sha is not None and actual_sha != str(expected_sha):
            raise RuntimeError(
                f"ERR_CHECKPOINT_CHECKSUM_MISMATCH: SHA-256 mismatch for '{rel_path}' "
                f"(expected {expected_sha}, got {actual_sha})"
            )


def _strip_model_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DDP ``module.`` and ``torch.compile`` ``_orig_mod.`` prefixes from state_dict keys."""
    cleaned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        k = key
        while k.startswith(("module.", "_orig_mod.")):
            if k.startswith("module."):
                k = k[len("module.") :]
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod.") :]
        cleaned[k] = val
    return cleaned


def _unwrap_raw_module(model: Any) -> Any:
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


# ---------------------------------------------------------------------------
# PPL (Perplexity) Computation
# ---------------------------------------------------------------------------

def compute_perplexity(loss: float, base: float = torch.e) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity is defined as exp(loss) for natural log, or 2^loss for log2.
    It measures how "surprised" the model is by the data – lower is better.
    For a perfect model (loss=0), perplexity = 1.

    Mathematical definition:
        PPL = exp(avg_negative_log_likelihood)

    Properties:
    - PPL ∈ [1, ∞)
    - PPL = 1 means perfect prediction (loss = 0)
    - PPL = vocab_size means random uniform guessing

    WHY: Perplexity is the standard metric for language modeling because it's
    interpretable (effective vocabulary size) and monotonic with loss.

    Parameters
    ----------
    loss : float
        Average cross-entropy loss (natural log base, typically from nn.CrossEntropyLoss).
    base : float
        Base of exponentiation. Default `torch.e` (natural log) matches PyTorch's loss.
        Use `base=2` for bits-per-character (BPC) style perplexity.

    Returns
    -------
    float
        Perplexity value (always >= 1.0).

    Example
    -------
        >>> loss = 2.3026  # ln(10)
        >>> compute_perplexity(loss)
        10.0  # Model is as surprised as guessing from 10 tokens

        >>> loss = 0.6931  # ln(2)
        >>> compute_perplexity(loss)
        2.0   # Effective vocabulary size = 2
    """
    import math
    if base == torch.e or base == math.e:
        return math.exp(loss)
    return base ** loss


def compute_perplexity_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[float, float]:
    """
    Compute both loss and perplexity from model logits in one pass.

    This is more efficient than calling compute_perplexity separately because
    it avoids recomputing the loss.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape (batch_size, seq_len, vocab_size).
    labels : torch.Tensor
        Ground truth labels of shape (batch_size, seq_len).
    ignore_index : int
        Label value to ignore in loss computation (typically -100 for padding).

    Returns
    -------
    tuple[float, float]
        (loss, perplexity) where loss is the average cross-entropy loss.

    Example
    -------
        >>> outputs = model(input_ids)
        >>> loss, ppl = compute_perplexity_from_logits(outputs.logits, labels)
        >>> print(f"Loss: {loss:.4f}, PPL: {ppl:.2f}")
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    # Reshape logits: (B, N, V) -> (B*N, V)
    # Reshape labels: (B, N) -> (B*N)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    ppl = compute_perplexity(loss.item())
    return loss.item(), ppl


def format_metrics_with_ppl(metrics: dict[str, float]) -> dict[str, float]:
    """
    Automatically add perplexity to metrics dict if loss is present.

    Convenience function for evaluation loops: if metrics contains "eval_loss"
    or "loss", add a "perplexity" field automatically.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing at least "loss" or "eval_loss".

    Returns
    -------
    dict[str, float]
        Original metrics with added "perplexity" key.

    Example
    -------
        >>> metrics = {"eval_loss": 1.234}
        >>> metrics = format_metrics_with_ppl(metrics)
        >>> print(metrics)
        {"eval_loss": 1.234, "perplexity": 3.43, "ppl": 3.43}
    """
    loss_key = "eval_loss" if "eval_loss" in metrics else "loss"
    if loss_key in metrics:
        ppl = compute_perplexity(metrics[loss_key])
        metrics["perplexity"] = ppl
        metrics["ppl"] = ppl
    return metrics


# ---------------------------------------------------------------------------
# Helper to determine if tokenizer needs saving
# ---------------------------------------------------------------------------

def _tokenizer_needs_saving(tokenizer_config: Optional[Any]) -> bool:
    """
    Determine if tokenizer files need to be copied to checkpoint.

    Returns True if:
    - tokenizer_type is "sentencepiece" (local file)
    - tokenizer_name_or_path is a local path (starts with . or /)
    - local_files_only is True (we can't rely on Hub)
    - tokenizer_type is "custom" (user-provided, may be local)

    Returns False if tokenizer is from HuggingFace Hub (e.g., "gpt2", "meta-llama/Llama-2-7b")
    """
    if tokenizer_config is None:
        return True  # save to be safe

    # SentencePiece always needs saving (local file)
    if getattr(tokenizer_config, "tokenizer_type", None) == "sentencepiece":
        return True

    # Custom tokenizer - always save to be safe
    if getattr(tokenizer_config, "tokenizer_type", None) == "custom":
        return True

    # Check if tokenizer_name_or_path is a local path
    name_or_path = getattr(tokenizer_config, "tokenizer_name_or_path", "")
    if name_or_path.startswith(('.', '/')) or '\\' in name_or_path:
        return True

    # HuggingFace Hub identifier (e.g., "gpt2", "meta-llama/Llama-2-7b")
    # These don't need to be saved - they can be downloaded
    return False


# ---------------------------------------------------------------------------
# Advanced checkpointing (HuggingFace‑style + training state)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: Any,  # PreTrainedModel from transformers (has save_pretrained)
    optimizer: torch.optim.Optimizer,
    scheduler: Any,  # can be LambdaLR or our LRScheduler
    scaler: Optional[torch.amp.GradScaler],
    state: TrainerState,
    callback_handler: CallbackHandler,
    output_dir: str,
    step: Optional[int] = None,
    tag: Optional[str] = None,
    early_stopping: Optional[EarlyStoppingCallback] = None,
    additional_state: Optional[dict[str, Any]] = None,
    training_args: Optional[Any] = None,   # <-- save hyperparameters
    **kwargs: Any,
) -> None:
    tokenizer: Optional[Any] = kwargs.pop("tokenizer", None)
    tokenizer_config: Optional[Any] = kwargs.pop("tokenizer_config", None)
    atomic: bool = kwargs.pop("atomic", True)
    """
    Save a complete training checkpoint with atomic write and config preservation.

    Checkpoint Design
    -----------------
    Each checkpoint is a self-contained directory containing:
    - Model weights in HuggingFace format (config.json + pytorch_model.bin)
    - Tokenizer files (if tokenizer needs saving - local/SP/custom)
    - Training state (optimizer, scheduler, scaler, TrainerState, callback states)
    - Optional early stopping state and hyperparameters (training_args.json)

    Tokenizer Saving Logic:
    - For HuggingFace Hub tokenizers (gpt2, meta-llama/...): only save config reference
    - For SentencePiece: copy .model file and save config
    - For custom tokenizers: attempt save_pretrained or copy files
    - For local path tokenizers: save full tokenizer

    Atomic Write (if `atomic=True`):
    - Saves to a temporary directory then renames to final location.
    - Prevents corrupted checkpoints if the process crashes mid‑write.
    - Rename is atomic on POSIX filesystems.

    Rank‑0 only: Only the main process saves checkpoints in distributed training.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model with `save_pretrained` method.
    optimizer : torch.optim.Optimizer
        Optimizer whose state_dict will be saved.
    scheduler : Any
        Scheduler with `state_dict` method.
    scaler : Optional[torch.amp.GradScaler]
        AMP scaler (if used).
    state : TrainerState
        Current training state.
    callback_handler : CallbackHandler
        Handler containing all callback states.
    output_dir : str
        Base directory for all checkpoints.
    step : Optional[int]
        Step number for numbered checkpoint naming.
    tag : Optional[str]
        Tag (e.g., "best", "final") – overrides step.
    early_stopping : Optional[EarlyStoppingCallback]
        Early stopping instance (state saved separately).
    additional_state : Optional[dict]
        Extra user state (e.g., random generator state).
    training_args : Optional[Any]
        Training arguments (dataclass or dict) – saved as JSON for reproducibility.
    tokenizer : Optional[Any]
        Tokenizer instance to save (if provided).
    tokenizer_config : Optional[Any]
        Tokenizer configuration for metadata and saving logic.
    atomic : bool
        If True, write to temporary directory then rename (safer).

    Edge Cases:
    - If both `step` and `tag` are None, saves as "checkpoint-latest".
    - Missing `save_pretrained` will raise AttributeError.
    - If `atomic=True` and renaming fails, the temporary directory is left behind
      (caller should clean up). We log an error but do not raise.
    - If tokenizer is None, tokenizer saving is skipped.

    Example Usage
    -------------
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     scaler=scaler,
        ...     state=state,
        ...     callback_handler=handler,
        ...     output_dir="./output",
        ...     step=global_step,
        ...     tag="best",
        ...     training_args=args,
        ...     tokenizer=tokenizer,
        ...     tokenizer_config=main_config.tokenizer,
        ... )
    """
    if not should_save_on_rank0(state):
        if is_distributed_initialized():
            dist.barrier()
        return

    os.makedirs(output_dir, exist_ok=True)

    # Determine checkpoint folder name
    if tag is not None:
        folder_name = f"{_CHECKPOINT_PREFIX}{tag}"
    elif step is not None:
        folder_name = f"{_CHECKPOINT_PREFIX}{step}"
    else:
        folder_name = f"{_CHECKPOINT_PREFIX}latest"

    final_path = os.path.join(output_dir, folder_name)

    if atomic:
        temp_path = tempfile.mkdtemp(dir=output_dir, prefix=".tmp_checkpoint_")
    else:
        temp_path = final_path
        os.makedirs(temp_path, exist_ok=True)

    try:
        # 1. Save unwrapped model in HuggingFace / safetensors format
        model_to_save = _unwrap_raw_module(model)
        model_to_save.save_pretrained(temp_path)

        # 2. Save tokenizer if provided
        if tokenizer is not None:
            needs_save = _tokenizer_needs_saving(tokenizer_config)

            if needs_save:
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(temp_path)
                    logger.info("Tokenizer saved to %s", temp_path)
                elif tokenizer_config and getattr(tokenizer_config, "tokenizer_type", None) == "sentencepiece":
                    model_path = getattr(tokenizer_config, "tokenizer_model_path", None) or getattr(
                        tokenizer_config, "tokenizer_name_or_path", None
                    )
                    if model_path and os.path.exists(model_path):
                        shutil.copy2(model_path, os.path.join(temp_path, "spm.model"))
                        if hasattr(tokenizer_config, "to_dict"):
                            with open(os.path.join(temp_path, _TOKENIZER_CONFIG_NAME), "w") as f:
                                json.dump(tokenizer_config.to_dict(), f, indent=2)
                        logger.info("SentencePiece tokenizer saved to %s", temp_path)
                else:
                    if tokenizer_config and hasattr(tokenizer_config, "to_dict"):
                        with open(os.path.join(temp_path, _TOKENIZER_CONFIG_NAME), "w") as f:
                            json.dump(tokenizer_config.to_dict(), f, indent=2)
                        logger.info("Tokenizer config saved (fallback)")
            else:
                if tokenizer_config and hasattr(tokenizer_config, "to_dict"):
                    with open(os.path.join(temp_path, _TOKENIZER_CONFIG_NAME), "w") as f:
                        json.dump(tokenizer_config.to_dict(), f, indent=2)
                    logger.info("Tokenizer config saved (Hub reference)")

        # 3. Save training hyperparameters (TrainingArguments.to_dict() first!)
        if training_args is not None:
            if hasattr(training_args, "to_dict") and callable(training_args.to_dict):
                args_dict = training_args.to_dict()
            elif hasattr(training_args, "__dataclass_fields__"):
                import dataclasses

                args_dict = dataclasses.asdict(training_args)
            elif isinstance(training_args, dict):
                args_dict = training_args
            else:
                args_dict = {"args": str(training_args)}
            with open(os.path.join(temp_path, "training_args.json"), "w") as f:
                json.dump(args_dict, f, indent=2)

        # 4. Build and save training state dictionary (including full RNG state)
        clean_trainer_state = {
            k: v for k, v in state.__dict__.items() if not k.startswith("_")
        }
        training_state: dict[str, Any] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            _KEY_TRAINER_STATE: clean_trainer_state,
            _KEY_CALLBACK_STATES: callback_handler.state_dict(),
            "rng_state": capture_rng_state(),
        }
        if early_stopping is not None:
            training_state["early_stopping"] = {
                "best_metric": early_stopping.best_metric,
                "patience_counter": early_stopping.early_stopping_patience_counter,
            }
        if additional_state:
            training_state["additional"] = additional_state

        torch.save(training_state, os.path.join(temp_path, "training_state.pt"))

        # 5. Write SHA-256 checkpoint_manifest.json before atomic commit
        write_checkpoint_manifest(temp_path, step=step, tag=tag)

        # 6. Atomic commit: replace final directory with temporary one
        if atomic:
            if os.path.exists(final_path):
                shutil.rmtree(final_path)
            shutil.move(temp_path, final_path)
            logger.info("Checkpoint saved atomically: %s", final_path)
        else:
            logger.info("Checkpoint saved: %s", final_path)

    except BaseException as e:
        logger.error("Failed to save checkpoint: %s", e)
        if atomic and os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
        raise

    if is_distributed_initialized():
        dist.barrier()


def load_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    state: TrainerState,
    callback_handler: CallbackHandler,
    checkpoint_path: str,
    device: torch.device,
    early_stopping: Optional[EarlyStoppingCallback] = None,
    tokenizer_config: Optional[Any] = None,
    restore_rng: bool = True,
) -> Any:
    """
    Restore complete training state (weights, optimizer, scheduler, scaler,
    TrainerState, callbacks, and full Python/NumPy/Torch/CUDA RNG) from disk
    after verifying SHA-256 checksums against ``checkpoint_manifest.json``.
    """
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist or is not a directory: {checkpoint_path}"
        )

    # Phase 0: Cryptographic integrity verification (SHA-256 manifest)
    verify_checkpoint_manifest(checkpoint_path)

    logger.info("Resuming from checkpoint: %s", checkpoint_path)

    # Phase 1: Restore model weights IN-PLACE so optimizer param_groups remain bound
    model_to_load = _unwrap_raw_module(model)
    safe_weights_path = os.path.join(checkpoint_path, "model.safetensors")
    bin_weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    raw_state_dict: Optional[dict[str, torch.Tensor]] = None
    if os.path.exists(safe_weights_path):
        from safetensors.torch import load_file as safetensors_load_file

        map_loc = str(device) if device is not None else "cpu"
        raw_state_dict = safetensors_load_file(safe_weights_path, device=map_loc)
        logger.info("Model weights read from model.safetensors")
    elif os.path.exists(bin_weights_path):
        raw_state_dict = torch.load(bin_weights_path, map_location=device, weights_only=True)
        logger.info("Model weights read from pytorch_model.bin")

    if raw_state_dict is not None:
        state_dict = _strip_model_prefixes(raw_state_dict)
        # Restore tied lm_head.weight if omitted by save_pretrained
        if (
            "lm_head.weight" not in state_dict
            and "wte.weight" in state_dict
            and hasattr(model_to_load, "lm_head")
        ):
            state_dict["lm_head.weight"] = state_dict["wte.weight"]
        model_to_load.load_state_dict(state_dict, strict=True)
        if hasattr(model_to_load, "_tie_weights"):
            model_to_load._tie_weights()
        loaded_model = model
    else:
        # Sharded HuggingFace fallback — copy state_dict in-place when compatible
        model_class = model_to_load.__class__
        tmp_loaded = model_class.from_pretrained(checkpoint_path)
        model_to_load.load_state_dict(tmp_loaded.state_dict(), strict=True)
        if hasattr(model_to_load, "_tie_weights"):
            model_to_load._tie_weights()
        loaded_model = model
        logger.info("Model loaded in-place via sharded from_pretrained fallback")

    if device is not None:
        loaded_model = loaded_model.to(device)

    # Phase 2: Restore training state & RNG
    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path, map_location=device, weights_only=True)

        if "optimizer" in training_state and optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])

        if (
            scheduler is not None
            and "scheduler" in training_state
            and training_state["scheduler"] is not None
        ):
            scheduler.load_state_dict(training_state["scheduler"])

        if scaler is not None and "scaler" in training_state and training_state["scaler"] is not None:
            scaler.load_state_dict(training_state["scaler"])

        saved_state = training_state.get(_KEY_TRAINER_STATE, {})
        for key, value in saved_state.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                logger.warning("Saved TrainerState has unknown key '%s', skipping.", key)

        if _KEY_CALLBACK_STATES in training_state:
            callback_handler.load_state_dict(training_state[_KEY_CALLBACK_STATES])

        if early_stopping is not None and "early_stopping" in training_state:
            es_state = training_state["early_stopping"]
            early_stopping.best_metric = es_state.get("best_metric", early_stopping.best_metric)
            early_stopping.early_stopping_patience_counter = es_state.get(
                "patience_counter", early_stopping.early_stopping_patience_counter
            )

        additional = training_state.get("additional")
        if isinstance(additional, dict) and "dataloader_state" in additional:
            object.__setattr__(state, "_resumed_dataloader_state", additional["dataloader_state"])

        if restore_rng and "rng_state" in training_state and training_state["rng_state"] is not None:
            restore_rng_state(training_state["rng_state"])

        logger.info("Training state restored: step=%d, epoch=%.2f", state.global_step, state.epoch)
    else:
        logger.warning("No training_state.pt found – starting from scratch (step=0).")

    # Phase 3: Restore tokenizer config if available
    tokenizer_config_path = os.path.join(checkpoint_path, _TOKENIZER_CONFIG_NAME)
    if os.path.exists(tokenizer_config_path) and tokenizer_config is not None:
        try:
            with open(tokenizer_config_path, "r") as f:
                saved_tokenizer_config = json.load(f)
            for key, value in saved_tokenizer_config.items():
                if hasattr(tokenizer_config, key):
                    setattr(tokenizer_config, key, value)
            logger.info("Tokenizer config updated from checkpoint")
        except Exception as e:
            logger.warning("Could not load tokenizer config: %s", e)

    return loaded_model


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Return the path to the most recent checkpoint in `output_dir`.

    WHY: Enables automatic resume from the latest checkpoint without manual
    path selection. The trainer can call this function to find the newest
    checkpoint by modification time.

    Sorting heuristic: Uses file modification time (most recent first).
    This is reliable even if step numbers are not monotonic or if checkpoints
    were manually copied.

    Edge Cases:
    - If the directory does not exist or contains no checkpoint folders,
      returns None.
    - Only directories starting with "checkpoint-" are considered.

    Performance: O(n log n) where n = number of checkpoint directories.
    Typically n <= 100, so negligible.

    Example Usage
    -------------
        >>> latest = get_latest_checkpoint("./output")
        >>> if latest:
        ...     load_checkpoint(..., checkpoint_path=latest, ...)
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith(_CHECKPOINT_PREFIX) and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)), reverse=True)
    return os.path.join(output_dir, checkpoints[0])


def prune_checkpoints(output_dir: str, save_total_limit: Optional[int]) -> None:
    """
    Remove the oldest periodic checkpoints when exceeding ``save_total_limit``.

    Pruning Strategy
    ----------------
    - Only numbered checkpoints (e.g., "checkpoint-1000") are pruned.
    - Tagged checkpoints ("best", "final", "epoch-3", etc.) are preserved.
    - Sorting by modification time (oldest first) ensures the earliest checkpoints
      are removed.

    Why preserve tagged checkpoints?
    - "best": optimal model by validation metric.
    - "final": model at training completion.
    - "epoch-N": epoch boundaries for analysis.
    - "early-stopped": state when early stopping triggered.

    Complexity: O(n log n) sorting, then O(k) deletions. For n ≤ 1000, acceptable.

    Edge Cases:
    - If `save_total_limit` is None or ≤ 0, no pruning.
    - If deletion fails due to permissions, logs a warning but continues.

    Example Usage
    -------------
        >>> prune_checkpoints("./output", save_total_limit=5)
        # Keeps only the 5 newest numbered checkpoints.
    """
    if save_total_limit is None or not os.path.exists(output_dir):
        return

    def is_prunable(name: str) -> bool:
        suffix = name.replace(_CHECKPOINT_PREFIX, "", 1)
        return suffix.isdigit()

    all_checkpoints = [
        d for d in os.listdir(output_dir)
        if d.startswith(_CHECKPOINT_PREFIX) and os.path.isdir(os.path.join(output_dir, d))
    ]
    prunable = [d for d in all_checkpoints if is_prunable(d)]
    # Sort by modification time, oldest first
    prunable.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))

    while len(prunable) > save_total_limit:
        oldest = prunable.pop(0)
        try:
            shutil.rmtree(os.path.join(output_dir, oldest))
            logger.info("Pruned old checkpoint: %s", oldest)
        except (OSError, PermissionError) as e:
            logger.warning("Could not prune %s: %s", oldest, e)


# ---------------------------------------------------------------------------
# Legacy / simple checkpoint helpers (for backward compatibility)
# ---------------------------------------------------------------------------

def save_checkpoint_simple(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    callback_handler: CallbackHandler,
    additional_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Simplified checkpoint saving (single .pt file). Prefer `save_checkpoint` above.

    This function exists for backward compatibility and non‑HF models.
    It saves all state into one file, which is simpler but less modular.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        _KEY_TRAINER_STATE: state.__dict__,
        _KEY_CALLBACK_STATES: callback_handler.state_dict(),
    }
    if hasattr(scheduler, "state_dict"):
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if additional_state:
        checkpoint.update(additional_state)

    torch.save(checkpoint, path)
    logger.info("Simple checkpoint saved to %s", path)


def load_checkpoint_simple(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainerState,
    callback_handler: CallbackHandler,
    map_location: Optional[torch.device | str] = None,
) -> None:
    """
    Simplified checkpoint loading (single .pt file). Prefer `load_checkpoint` above.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    # Model
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])

    # Optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Scheduler
    if "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Trainer state
    saved_state = checkpoint[_KEY_TRAINER_STATE]
    for key, value in saved_state.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # Callback handler states
    if "callback_states" in checkpoint:
        callback_handler.load_state_dict(checkpoint["callback_states"])

    logger.info("Simple checkpoint loaded from %s (step %d)", path, state.global_step)


# ---------------------------------------------------------------------------
# Scheduling helpers (logging, evaluation, saving)
# ---------------------------------------------------------------------------

def should_log(
    state: TrainerState,
    logging_steps: int,
    log_on_each_step: bool = False,
) -> bool:
    """
    Determine whether the current step should trigger logging.

    WHY: Logging every step may be too verbose; this helper implements the
    common pattern: log if `global_step % logging_steps == 0` (or at step 0).

    Edge Cases:
    - Step 0 is always logged (except when `logging_steps <= 0` and `log_on_each_step` is False).
    - If `logging_steps` is 1, every step triggers logging.
    - Negative `logging_steps` disables logging completely.
    """
    if logging_steps <= 0:
        return False
    if log_on_each_step:
        return True
    return state.global_step % logging_steps == 0 or state.global_step == 0


def should_evaluate(
    state: TrainerState,
    eval_steps: int,
    eval_on_each_epoch: bool = False,
    epoch: Optional[float] = None,
) -> bool:
    """
    Determine whether evaluation should be performed at this point (step‑based only).

    NOTE: Epoch‑based evaluation must be handled by the trainer at epoch boundaries.
    This helper only returns True for step‑based evaluation (`global_step % eval_steps == 0`).
    """
    if eval_steps <= 0:
        return False
    return state.global_step % eval_steps == 0 and state.global_step > 0


def should_save(state: TrainerState, save_steps: int, save_on_each_epoch: bool = False) -> bool:
    """Determine whether a checkpoint should be saved (step‑based only)."""
    if save_steps <= 0:
        return False
    return state.global_step % save_steps == 0 and state.global_step > 0


# ---------------------------------------------------------------------------
# Loss and gradient utilities
# ---------------------------------------------------------------------------

def average_losses(loss_dict: dict[str, float], steps: int = 1) -> dict[str, float]:
    """
    Average a dictionary of accumulated losses over a number of steps.

    WHY: When using gradient accumulation, losses are often summed over micro‑batches.
    This helper divides each loss by the number of accumulation steps to obtain
    the per‑step average, which is more meaningful for logging.
    """
    if steps <= 1:
        return loss_dict.copy()
    return {k: v / steps for k, v in loss_dict.items()}


def clip_grad_norm_(model: torch.nn.Module, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """Clip gradients of the model to a maximum norm. Handles DDP wrapping."""
    model_to_clip = model.module if hasattr(model, "module") else model
    return torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), max_norm, norm_type)


# ---------------------------------------------------------------------------
# Learning rate helpers
# ---------------------------------------------------------------------------

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate of the first parameter group."""
    if not optimizer.param_groups:
        return 0.0
    return optimizer.param_groups[0]["lr"]


def get_lr_group_info(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return a list of dictionaries containing learning rate and weight decay per group."""
    return [
        {"lr": group["lr"], "weight_decay": group.get("weight_decay", 0.0)}
        for group in optimizer.param_groups
    ]