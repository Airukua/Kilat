from __future__ import annotations
import logging
import os
import re
from typing import Any, Optional

from .callbacks import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _sanitize_key(key: str) -> str:
    """
    Normalise a metric key for backends that reject special characters.

    WHY: Different logging backends have different restrictions on metric keys.
    TensorBoard accepts slashes for grouping, W&B handles most characters, but
    MLflow and Comet can choke on `/`, spaces, or punctuation. This function
    provides a conservative normalisation that works everywhere.

    Rules applied (in order):
    1. Replace `/` and space with `_` (creates valid identifier).
    2. Strip any character that is not alphanumeric, underscore, hyphen, or dot.
       Hyphens are kept because some metrics use "f1-score".
    3. Collapse consecutive underscores into a single one.
    4. Strip leading/trailing underscores.

    Edge Cases / Trade-offs:
    - Dot (`.`) is preserved: it's rarely a problem and useful for nested
      structures (e.g., "loss.value").
    - This conversion is lossy: "eval/loss" and "eval_loss" become the same key.
      That's acceptable because logging backends treat them as distinct metrics
      anyway; we just avoid the backend's rejection.

    Performance: O(n) where n = length of key; called once per metric per log step.
    For 100 metrics logged every 10 steps, overhead is negligible.

    Example Usage
    -------------
        >>> _sanitize_key("eval/loss")        # "eval_loss"
        >>> _sanitize_key("train loss")       # "train_loss"
        >>> _sanitize_key("f1-score")         # "f1-score"
        >>> _sanitize_key("attention@layer1") # "attention_layer1"
    """
    key = key.replace("/", "_").replace(" ", "_")
    key = re.sub(r"[^\w\-\.]", "_", key)
    key = re.sub(r"_+", "_", key)
    return key.strip("_")


def _is_package_available(package_name: str) -> bool:
    """
    Return True if ``package_name`` can be imported.

    WHY: We need to check optional dependencies without actually importing them
    (which would cause unnecessary overhead and potential side effects). Using
    `importlib.util.find_spec` is the standard, lightweight way.

    Assumptions: The package is installed in a location discoverable by Python's
    import system. It doesn't work for namespace packages that have no __path__,
    but those are rare and not relevant here.

    Performance: O(1) after the first call per package (module spec is cached).

    Example Usage
    -------------
        >>> _is_package_available("torch")
        True
        >>> _is_package_available("nonexistent")
        False
    """
    import importlib.util
    return importlib.util.find_spec(package_name) is not None


# ---------------------------------------------------------------------------
# Abstract base for all integrations
# ---------------------------------------------------------------------------

class IntegrationCallback(TrainerCallback):
    """
    Base class for logging-backend integrations.

    WHY: Provides a template for plugging external logging systems (TensorBoard,
    W&B, MLflow, Comet, etc.) into the trainer's lifecycle. Subclasses must
    implement the core methods that define the integration's behaviour.

    Subclassing Contract:
    ---------------------
    - `is_available()`: classmethod → bool. Returns True iff the backend library
      is installed. Called before instantiation.
    - `on_train_begin()`: initialise the run (create experiment, set up logging).
    - `on_log()`: write a metrics dict to the backend.
    - `on_train_end()`: close/finalise the run.

    All other hooks (on_evaluate, on_save, etc.) are optional overrides.

    Helper for subclasses:
    ----------------------
    `_should_log(state)`: Returns True only for the main process in distributed
    training. Prevents duplicate logging from every worker.

    Design Note:
    ------------
    The base class does NOT implement `__init__` because subclasses have varying
    parameter needs (e.g., `watch_model` for Wandb, `flatten_params` for MLflow).
    The trainer instantiates via the factory function `get_reporting_integration_callbacks`
    which passes no arguments; subclasses must provide default values for all
    parameters in their constructors.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the backend library is installed and usable."""
        raise NotImplementedError

    def _should_log(self, state: TrainerState) -> bool:
        """
        Return True only for the main process in distributed training.

        WHY: In distributed setups, only the process with rank 0 (world_process_zero)
        should log to avoid duplicate entries and API rate limits.
        """
        return state.is_world_process_zero


# ---------------------------------------------------------------------------
# TensorBoard
# ---------------------------------------------------------------------------

class TensorBoardCallback(IntegrationCallback):
    """
    Logs training metrics to TensorBoard.

    Why TensorBoard:
    - de facto standard for visualising scalar metrics, histograms, and graphs.
    - No network dependency, writes to local disk.
    - Integrates with PyTorch's native `SummaryWriter`.

    Behaviour:
    ----------
    - Each scalar in the `logs` dict is written as a separate scalar summary
      at `global_step`.
    - Nested dicts are flattened with `/` as separator (e.g., `{"eval": {"loss": 0.5}}`
      → `"eval/loss"`). TensorBoard interprets `/` as grouping in its UI.
    - The SummaryWriter is created at `on_train_begin` using `args.output_dir`
      as the log directory (same directory as checkpoints) unless
      `TENSORBOARD_LOG_DIR` env var overrides it.

    External Dependencies:
    ---------------------
    - Tries `from torch.utils.tensorboard import SummaryWriter` first (PyTorch ≥1.2).
    - Falls back to `tensorboardX` if the torch version is too old or missing.
    - Requires `tensorboard` package to actually view logs (`pip install tensorboard`),
      but the writer works without it.

    Edge Cases & Assumptions:
    -------------------------
    - If `tb_writer` is passed in constructor, it's used as-is; otherwise created.
    - `_should_log` ensures only rank 0 writes.
    - The writer is flushed after each `on_log` call to ensure data is persisted
      even if training crashes immediately after.
    - The writer is closed in `on_train_end`; calling `close()` twice is harmless.

    Performance:
    -----------
    - `add_scalar` is cheap (O(1) per call). Flushing every log step adds disk I/O,
      but TensorBoard's writer buffers internally; explicit flush guarantees data
      safety at the cost of some performance (~10‑50 µs per call). Acceptable.

    Example Usage
    -------------
        >>> callback = TensorBoardCallback()
        >>> # Or with pre‑created writer:
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> writer = SummaryWriter("my_logs")
        >>> callback = TensorBoardCallback(tb_writer=writer)
    """

    def __init__(self, tb_writer=None) -> None:
        self._tb_writer = tb_writer

    @classmethod
    def is_available(cls) -> bool:
        return _is_package_available("tensorboard") or _is_package_available(
            "tensorboardX"
        )

    def _init_writer(self, log_dir: str) -> None:
        """Lazy-import and create the SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter  # type: ignore[no-redef]
        self._tb_writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging to: %s", log_dir)

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return
        if self._tb_writer is None:
            log_dir = os.environ.get(
                "TENSORBOARD_LOG_DIR",
                getattr(args, "logging_dir", None) or getattr(args, "output_dir", "./runs"),
            )
            self._init_writer(log_dir)

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state) or self._tb_writer is None or logs is None:
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self._tb_writer.add_scalar(key, value, global_step=state.global_step)
            elif isinstance(value, dict):
                # Flatten one level of nesting: {"eval": {"loss": 0.5}} → "eval/loss"
                # TensorBoard uses '/' for grouping in its UI.
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self._tb_writer.add_scalar(
                            f"{key}/{sub_key}", sub_value, global_step=state.global_step
                        )
        self._tb_writer.flush()   # Ensure data written even if crash occurs.

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
            logger.info("TensorBoard writer closed.")


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

class WandbCallback(IntegrationCallback):
    """
    Logs training metrics, hyperparameters, and system info to W&B.

    Why W&B:
    - Centralised experiment tracking with rich UI, comparisons, and collaboration.
    - Supports automatic logging of system metrics (CPU, GPU, memory).
    - Hyperparameter sweeps and model versioning.

    Run initialisation
    ------------------
    A W&B run is created at `on_train_begin` if one is not already active.
    The run is named from `args.run_name` (or auto-generated by W&B) and
    the project from the `WANDB_PROJECT` env var (default: `"trainer"`).

    Hyperparameters
    ---------------
    All fields of `args` that are JSON-serialisable scalars are logged as
    W&B config at the start of the run. This captures learning rate, batch
    size, scheduler type, etc. without any extra code in the trainer.

    Gradient / weight histograms
    ----------------------------
    Pass `watch_model=True` to log model weight and gradient histograms via
    `wandb.watch()`. Disabled by default because it adds significant
    overhead per step and is rarely needed outside of debugging.

    Environment variables recognised
    ---------------------------------
    WANDB_PROJECT   : W&B project name (default: "trainer")
    WANDB_ENTITY    : W&B entity (team or username)
    WANDB_RUN_NAME  : Override for the run display name
    WANDB_DISABLED  : Set to "true" to silence W&B entirely (useful in CI)

    Assumptions & Edge Cases:
    --------------------------
    - If `wandb.run` already exists (e.g., from a notebook cell), the callback
      will reuse it instead of creating a new run. This avoids nested runs.
    - `wandb.watch(model)` must be called after the model has moved to the device.
      The callback receives `model` via `on_train_begin(..., model=model)`.
    - W&B can be disabled by setting `WANDB_DISABLED=true`; the callback will
      silently skip logging (no errors).
    - The callback logs the full `logs` dict as-is – W&B accepts nested dicts.

    Performance:
    ------------
    - `wandb.log()` is asynchronous by default, so it doesn't block training.
    - `wandb.watch()` adds overhead per step: histograms of weights/gradients are
      computed. Set `watch_log_freq` to a high value (e.g., 1000) to reduce cost.
    - Disabling `watch_model` is recommended for large models (e.g., >1B params).

    Example Usage
    -------------
        >>> callback = WandbCallback(watch_model=True, watch_log_freq=500)
        >>> # In trainer:
        >>> handler.add_callback(callback)
    """

    def __init__(
        self,
        watch_model: bool = False,
        watch_log_freq: int = 100,
    ) -> None:
        self.watch_model = watch_model
        self.watch_log_freq = watch_log_freq
        self._wandb = None
        self._run = None
        self._initialized = False

    @classmethod
    def is_available(cls) -> bool:
        return _is_package_available("wandb")

    def _setup(self, args: Any, state: TrainerState, model=None) -> None:
        """Initialise the W&B run and optionally watch the model."""
        import wandb  # noqa: PLC0415

        self._wandb = wandb

        if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
            logger.info("W&B disabled via WANDB_DISABLED env var.")
            return

        # Collect hyperparameters from args dataclass / plain object.
        hp_config: dict[str, Any] = {}
        if hasattr(args, "__dataclass_fields__"):
            import dataclasses
            hp_config = {
                k: v for k, v in dataclasses.asdict(args).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }

        project = os.environ.get("WANDB_PROJECT", "trainer")
        entity = os.environ.get("WANDB_ENTITY", None)
        run_name = (
            os.environ.get("WANDB_RUN_NAME")
            or getattr(args, "run_name", None)
            or None
        )

        # Reuse existing run if one is already active (e.g. notebook usage).
        if wandb.run is None:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=hp_config,
                resume="allow",
                tags=getattr(args, "wandb_tags", None),
            )
        else:
            self._run = wandb.run
            self._run.config.update(hp_config, allow_val_change=True)

        if self.watch_model and model is not None:
            wandb.watch(model, log="all", log_freq=self.watch_log_freq)

        self._initialized = True
        logger.info("W&B run initialised: %s", self._run.url if self._run else "unknown")

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return
        self._setup(args, state, model=model)

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if (
            not self._should_log(state)
            or not self._initialized
            or self._wandb is None
            or logs is None
        ):
            return
        # W&B accepts nested dicts natively; no flattening needed.
        # Adding train/global_step makes it easy to compare runs by step.
        self._wandb.log({**logs, "train/global_step": state.global_step})

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._initialized and self._wandb is not None:
            self._wandb.finish()
            self._initialized = False
            logger.info("W&B run finished.")

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a summary of the best metric so far to the W&B run summary."""
        if (
            not self._should_log(state)
            or not self._initialized
            or self._wandb is None
            or metrics is None
        ):
            return
        if self._run is not None and state.best_metric is not None:
            self._run.summary["best_metric"] = state.best_metric


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

class MLflowCallback(IntegrationCallback):
    """
    Logs training metrics and parameters to MLflow Tracking.

    Why MLflow:
    - Open‑source platform for the complete ML lifecycle.
    - Supports local file‑based tracking (`./mlruns`) and remote servers.
    - Parameter and metric versioning, model registry.

    Run management
    --------------
    A new MLflow run is created at `on_train_begin`.  If an active run
    already exists in the current context (e.g. nested runs or notebook
    usage), the callback will log into that run instead of creating a new one.
    This is intentional: it allows the trainer to be part of a larger experiment.

    Tracking URI
    ------------
    Set `MLFLOW_TRACKING_URI` env var to point at a remote MLflow server.
    Defaults to a local `./mlruns` directory.

    Metric key constraints
    ----------------------
    MLflow metric keys must match `[A-Za-z0-9._\- /]` and be ≤ 250 chars.
    Keys are sanitised via `_sanitize_key()` before logging.

    Assumptions & Edge Cases:
    --------------------------
    - Parameters are logged only if `flatten_params=True` (default). MLflow params
      are immutable after first log, so this is disabled for resumed runs to avoid
      conflicts. The flag exists to let the user decide.
    - Metric values are cast to `float`; non‑numeric logs are skipped.
    - The best metric is logged as a final metric and also stored as a tag for
      the best checkpoint path.
    - MLflow runs are ended in `on_train_end`; closing a run that is already ended
      is a no‑op.

    Performance:
    ------------
    - Logging a metric is an HTTP call to the tracking server if remote;
      use `flush_period` or batch logging for high‑frequency logging.
    - Local file logging is fast (microseconds per call).

    Example Usage
    -------------
        >>> callback = MLflowCallback(flatten_params=True)
        >>> # Use with remote server:
        >>> os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.example.com"
    """

    def __init__(self, flatten_params: bool = True) -> None:
        self.flatten_params = flatten_params
        self._ml = None
        self._run = None

    @classmethod
    def is_available(cls) -> bool:
        return _is_package_available("mlflow")

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return

        import mlflow  # noqa: PLC0415

        self._ml = mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            getattr(args, "run_name", "trainer_experiment"),
        )
        mlflow.set_experiment(experiment_name)

        # Re-use active run if already inside one (e.g. parent run in notebooks).
        if mlflow.active_run() is None:
            self._run = mlflow.start_run(
                run_name=getattr(args, "run_name", None)
            )
        else:
            self._run = mlflow.active_run()

        if self.flatten_params and hasattr(args, "__dataclass_fields__"):
            import dataclasses
            params = {
                _sanitize_key(k): str(v)
                for k, v in dataclasses.asdict(args).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            # MLflow param values must be strings and ≤ 500 chars.
            params = {k: v[:500] for k, v in params.items()}
            mlflow.log_params(params)

        logger.info(
            "MLflow run started: %s (experiment: %s)",
            self._run.info.run_id,
            experiment_name,
        )

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if (
            not self._should_log(state)
            or self._ml is None
            or logs is None
        ):
            return
        scalar_metrics = {
            _sanitize_key(k): float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float))
        }
        if scalar_metrics:
            self._ml.log_metrics(scalar_metrics, step=state.global_step)

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._ml is not None and self._run is not None:
            # Log best metric as a final summary param.
            if state.best_metric is not None:
                self._ml.log_metric("best_metric", state.best_metric)
            if state.best_model_checkpoint is not None:
                self._ml.set_tag(
                    "best_model_checkpoint", state.best_model_checkpoint
                )
            self._ml.end_run()
            self._run = None
            logger.info("MLflow run ended.")


# ---------------------------------------------------------------------------
# Comet ML
# ---------------------------------------------------------------------------

class CometMLCallback(IntegrationCallback):
    """
    Logs training metrics and hyperparameters to Comet ML.

    Why Comet:
    - Enterprise‑focused experiment tracking with extensive visualisations.
    - Integrated model registry and production monitoring.
    - Supports hyperparameter sweeps and advanced comparisons.

    Authentication
    --------------
    Comet authenticates via the `COMET_API_KEY` env var or the
    `~/.comet.config` file.  If neither is present, initialisation will
    fail with a descriptive error. The callback logs a warning and skips
    logging if the API key is missing (trainer continues).

    Experiment naming
    -----------------
    Project name: `COMET_PROJECT_NAME` env var (default: `"general"`).
    Workspace:    `COMET_WORKSPACE` env var (optional).
    Experiment name: `args.run_name` if available.

    Metric key normalisation
    ------------------------
    Comet accepts most key formats, but slashes and special characters can
    cause UI rendering issues.  Keys are sanitised via `_sanitize_key()`.

    Assumptions & Edge Cases:
    --------------------------
    - If a pre‑existing `experiment` is provided via constructor, it is used
      directly; otherwise a new experiment is created.
    - Comet experiments are ended in `on_train_end`; if the experiment was provided
      externally, ending it may interfere with other code. The callback ends it anyway
      because that's the expected behaviour for a callback‑managed lifecycle.
    - The `best_metric` is logged as a separate metric after training finishes.

    Performance:
    ------------
    - `log_metric` is asynchronous by default, so it does not block training.
    - Each call creates an HTTP request; batch logging is not supported.
      For high‑frequency logging (e.g., every step), consider reducing `log_every_n_steps`.

    Example Usage
    -------------
        >>> callback = CometMLCallback()
        >>> # With custom experiment:
        >>> from comet_ml import Experiment
        >>> exp = Experiment(api_key="...", project_name="my_project")
        >>> callback = CometMLCallback(experiment=exp)
    """

    def __init__(self, experiment=None) -> None:
        self._experiment = experiment
        self._comet = None

    @classmethod
    def is_available(cls) -> bool:
        return _is_package_available("comet_ml")

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return

        import comet_ml  

        self._comet = comet_ml

        if self._experiment is None:
            api_key = os.environ.get("COMET_API_KEY")
            if not api_key:
                logger.warning(
                    "CometMLCallback: COMET_API_KEY not set. "
                    "Comet logging will be skipped."
                )
                self._comet = None
                return

            project_name = os.environ.get("COMET_PROJECT_NAME", "general")
            workspace = os.environ.get("COMET_WORKSPACE", None)
            self._experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=project_name,
                workspace=workspace,
            )

        # Set experiment name from run_name if available.
        run_name = getattr(args, "run_name", None)
        if run_name:
            self._experiment.set_name(run_name)

        # Log hyperparameters as Comet "parameters".
        if hasattr(args, "__dataclass_fields__"):
            import dataclasses
            hp = {
                k: v
                for k, v in dataclasses.asdict(args).items()
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            self._experiment.log_parameters(hp)

        logger.info(
            "Comet ML experiment started: %s",
            self._experiment.get_key(),
        )

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if (
            not self._should_log(state)
            or self._experiment is None
            or logs is None
        ):
            return
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self._experiment.log_metric(
                    _sanitize_key(key), value, step=state.global_step
                )

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._experiment is not None:
            if state.best_metric is not None:
                self._experiment.log_metric("best_metric", state.best_metric)
            self._experiment.end()
            self._experiment = None
            logger.info("Comet ML experiment ended.")


# ---------------------------------------------------------------------------
# Console / progress logger (always-on, no external dependency)
# ---------------------------------------------------------------------------

class ConsoleCallback(IntegrationCallback):
    """
    Logs training progress and validation results to the Python ``logging`` module.

    WHY: Provides immediate, human‑readable output in the terminal/logs without
    any external dependencies. Essential for debugging and for users who don't
    want to set up a tracking backend.

    Behaviour
    ---------
    - `on_log`: logs every scalar entry in the `logs` dict at INFO level, formatted as
      `step=N | loss=1.234 | ppl=3.45 | lr=3e-05 | epoch=1.00`.
    - `on_evaluate`: logs validation metrics as a dedicated block so they are easy
      to spot in plain console output.
    - `on_train_begin`: logs total steps and epoch count.
    - `on_train_end`: logs a training complete summary with best metric and checkpoint.

    Assumptions:
    ------------
    - The logging module is already configured (e.g., basicConfig called).
    - This callback is always available (`is_available() = True`).
    - It's included in `DEFAULT_CALLBACKS` so that even a bare trainer produces output.

    Trade-offs:
    -----------
    - Does NOT log to any external system, only to Python's log stream.
    - Not suitable for automated experiment comparison across runs.
    - The output format is fixed; users cannot easily customise it without subclassing.

    Performance:
    ------------
    - Logging is cheap but can slow down training if logging every step with many metrics.
      Set `log_every_n_steps` to a reasonable value (e.g., 10) in the trainer.

    Example Usage
    -------------
        >>> callback = ConsoleCallback()
        >>> # Automatically added via DEFAULT_CALLBACKS; no need to instantiate manually.
    """

    @classmethod
    def is_available(cls) -> bool:
        return True  # Always available — uses stdlib logging only.

    def on_train_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return
        logger.info(
            "Training started | total_steps=%d | epochs=%d",
            state.max_steps,
            state.num_train_epochs,
        )

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state) or logs is None:
            return
        # Keep scalar metrics in a deterministic order so logs are easy to scan.
        parts = [f"step={state.global_step}"]
        preferred_order = ["loss", "ppl", "lr", "grad_scale", "epoch"]
        seen: set[str] = set()
        for key in preferred_order:
            if key in logs and isinstance(logs[key], (int, float)):
                value = logs[key]
                parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
                seen.add(key)
        if "ppl" in seen and "perplexity" in logs:
            seen.add("perplexity")
        if "ppl" not in seen and "perplexity" in logs and isinstance(logs["perplexity"], (int, float)):
            value = logs["perplexity"]
            parts.append(f"ppl={value:.4g}" if isinstance(value, float) else f"ppl={value}")
            seen.add("perplexity")
        for key, value in logs.items():
            if key in seen:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            elif isinstance(value, int):
                parts.append(f"{key}={value}")
        logger.info(" | ".join(parts))

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state) or not metrics:
            return
        parts = [f"step={state.global_step}"]
        preferred_order = [
            "eval_loss",
            "ppl",
            "accuracy",
            "f1",
            "precision",
            "recall",
        ]
        seen: set[str] = set()
        for key in preferred_order:
            if key in metrics and isinstance(metrics[key], (int, float)):
                value = metrics[key]
                parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
                seen.add(key)
        if "ppl" in seen and "perplexity" in metrics:
            seen.add("perplexity")
        if "ppl" not in seen and "perplexity" in metrics and isinstance(metrics["perplexity"], (int, float)):
            value = metrics["perplexity"]
            parts.append(f"ppl={value:.4g}" if isinstance(value, float) else f"ppl={value}")
            seen.add("perplexity")
        for key, value in metrics.items():
            if key in seen:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            elif isinstance(value, int):
                parts.append(f"{key}={value}")
        logger.info("Validation | %s", " | ".join(parts))

    def on_train_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._should_log(state):
            return
        logger.info(
            "Training complete | total_steps=%d | best_metric=%s | checkpoint=%s",
            state.global_step,
            f"{state.best_metric:.6f}" if state.best_metric is not None else "n/a",
            state.best_model_checkpoint or "n/a",
        )


# Backwards-compatible alias for older imports and docs.
ProgressCallback = ConsoleCallback


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

# Maps report_to string → integration class.
_INTEGRATION_REGISTRY: dict[str, type[IntegrationCallback]] = {
    "tensorboard": TensorBoardCallback,
    "wandb": WandbCallback,
    "mlflow": MLflowCallback,
    "comet_ml": CometMLCallback,
}

# Always active regardless of report_to.
DEFAULT_CALLBACKS: list[TrainerCallback] = [ConsoleCallback()]


def get_reporting_integration_callbacks(
    report_to: list[str] | str | None,
) -> list[IntegrationCallback]:
    """
    Instantiate and return integration callbacks for the requested backends.

    WHY: This function is the single entry point for the trainer to set up logging
    integrations. It handles the complexity of:
    - Parsing the `report_to` argument (string, list, "all", "none").
    - Checking availability of each backend's library.
    - Falling back gracefully when a backend is requested but not installed.
    - Returning a list of ready‑to‑use callback instances.

    The trainer can simply do:
        callbacks = get_reporting_integration_callbacks(args.report_to)
        handler = CallbackHandler(callbacks + DEFAULT_CALLBACKS, args)

    Parameter interpretation:
    -------------------------
    - `"all"` → every registered integration that is available.
    - `"none"` or `None` → empty list (no external logging).
    - A list of strings → only the requested backends.
    - A single string → treated as a list with one element.

    Assumptions & Edge Cases:
    --------------------------
    - If a backend is requested but its library is not installed, a warning is logged
      and it is skipped (no exception). This allows the same code to run in different
      environments with different optional dependencies.
    - The `ConsoleCallback` is NOT included here – it's added separately via
      `DEFAULT_CALLBACKS` because it has no external dependency and should always
      be present.
    - Backend names are case‑insensitive (normalised to lower case).
    - Order of callbacks in the returned list is the same as the order in `report_to`
      (or the registry order for `"all"`).

    Performance:
    ------------
    - Each `is_available()` call performs an import‑lib check (cached). Called once
      per requested backend; negligible overhead.

    Example Usage
    -------------
        >>> # Enable all available backends
        >>> callbacks = get_reporting_integration_callbacks("all")
        >>> # Enable specific backends
        >>> callbacks = get_reporting_integration_callbacks(["wandb", "tensorboard"])
        >>> # Disable all external logging
        >>> callbacks = get_reporting_integration_callbacks("none")
    """
    if report_to is None or report_to == "none":
        return []

    if isinstance(report_to, str):
        report_to = [report_to]

    # Expand "all" without dropping explicitly requested backends.
    normalized_report_to = [
        name.strip().lower()
        for name in report_to
        if name.strip().lower() not in {"all", "none"}
    ]
    if not normalized_report_to:
        return []
    if any(name.strip().lower() == "all" for name in report_to):
        for name in _INTEGRATION_REGISTRY.keys():
            if name not in normalized_report_to:
                normalized_report_to.append(name)
    report_to = normalized_report_to

    unavailable_backends: list[str] = []
    unknown_backends: list[str] = []
    callbacks: list[IntegrationCallback] = []
    for name in report_to:
        name = name.strip().lower()
        if name not in _INTEGRATION_REGISTRY:
            unknown_backends.append(name)
            continue
        cls = _INTEGRATION_REGISTRY[name]
        if not cls.is_available():
            unavailable_backends.append(name)
            continue
        callbacks.append(cls())
        logger.info("Integration enabled: %s", name)

    if unknown_backends:
        logger.warning(
            "Unknown integration(s) requested: %s. Available: %s",
            unknown_backends,
            list(_INTEGRATION_REGISTRY.keys()),
        )
    if unavailable_backends:
        logger.warning(
            "Optional logging integration(s) not installed: %s. Training will "
            "continue without them. Install with: pip install %s",
            unavailable_backends,
            " ".join(unavailable_backends),
        )

    return callbacks


def register_integration(
    name: str,
    integration_cls: type[IntegrationCallback],
) -> None:
    """
    Register a custom integration so it is accessible via
    `get_reporting_integration_callbacks()`.

    WHY: Allows third‑party code to add new logging backends without modifying
    this file (open/closed principle). The trainer remains agnostic to which
    integrations exist.

    Parameters
    ----------
    name : str
        Unique string identifier (e.g. `"neptune"`).
    integration_cls : type[IntegrationCallback]
        A class that inherits from `IntegrationCallback` and implements
        at minimum `is_available()`, `on_train_begin`, `on_log`,
        and `on_train_end`.

    Raises
    ------
    TypeError  : `integration_cls` does not subclass `IntegrationCallback`.
    ValueError : `name` is already registered (either built‑in or custom).

    Side Effects:
    -------------
    Modifies the global registry `_INTEGRATION_REGISTRY`. After registration,
    `get_reporting_integration_callbacks([name])` will instantiate the custom callback.

    Example
    -------
        >>> class NeptuneCallback(IntegrationCallback):
        ...     @classmethod
        ...     def is_available(cls): return _is_package_available("neptune")
        ...     def on_train_begin(self, args, state, control, **kw):
        ...         import neptune
        ...         self.run = neptune.init_run()
        ...     def on_log(self, args, state, control, logs=None, **kw):
        ...         for k, v in logs.items():
        ...             if isinstance(v, (int, float)):
        ...                 self.run[k].log(v, step=state.global_step)
        ...     def on_train_end(self, args, state, control, **kw):
        ...         self.run.stop()
        ...
        >>> register_integration("neptune", NeptuneCallback)
        >>> # Now "neptune" can be used in report_to
    """
    if not (
        isinstance(integration_cls, type)
        and issubclass(integration_cls, IntegrationCallback)
    ):
        raise TypeError(
            f"integration_cls must subclass IntegrationCallback, "
            f"got {integration_cls}."
        )
    if name in _INTEGRATION_REGISTRY:
        raise ValueError(
            f"Integration '{name}' is already registered. "
            f"Use a different name or remove the existing registration first."
        )
    _INTEGRATION_REGISTRY[name] = integration_cls
    logger.info(
        "Registered custom integration: '%s' → %s", name, integration_cls.__name__
    )
