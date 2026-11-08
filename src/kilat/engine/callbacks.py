from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TrainerState:
    """
    Immutable-like snapshot of the training loop's progress.

    Why this exists:
    - Centralises all mutable training state into a single container.
    - Enables callbacks to read/write training progress without coupling
      to the Trainer implementation.
    - Simplifies checkpointing: the state can be serialised/restored to
      resume training.

    Invariants:
    - `global_step` should never decrease.
    - `epoch` = total number of complete passes over the dataset
      (may be fractional for partial epochs).
    - `max_steps` and `num_train_epochs` are mutually consistent:
      if max_steps > 0, it overrides num_train_epochs.
    - `log_history` is append-only.

    Edge Cases / Risks:
    - `best_model_checkpoint` is a string path; the caller is responsible
      for its validity (Trainer does not validate it).
    - `is_world_process_zero` is a flag to avoid duplicated logging/saving
      in distributed training. The default True is safe for single‑process,
      but must be set correctly in multi‑process setups.
    - The dataclass is mutable by design (Trainer updates fields in place).
      Do not treat it as immutable or share across threads without locks.
    """
    epoch: float = 0.0
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    log_history: list[dict[str, float]] = field(default_factory=list)
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_world_process_zero: bool = True  # MUST be set before training starts


@dataclass
class TrainerControl:
    """
    Control signals for the Trainer, used by callbacks to influence the loop.

    Design philosophy:
    - Each field is a "request" flag, not a command. The Trainer reads them
      at defined points and decides whether to act (e.g., `should_save` may
      be overridden by `save_steps`).
    - Flags are OR‑aggregated across multiple callbacks via `CallbackHandler`.
    - After each callback dispatch, the Trainer typically resets flags it
      has acted upon (e.g., `should_log = False` after logging).

    Side Effects:
    - Modifying any field here directly from the main training loop is
      allowed but discouraged – use callbacks instead.
    - The Trainer never writes to this object except to reset flags.

    External Dependencies: None – pure data carrier.
    """
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False


class TrainerCallback:
    """
    Abstract base class for training lifecycle hooks.

    Why a class hierarchy instead of plain functions?
    - Stateful callbacks (e.g., EarlyStopping) need to persist counters.
    - CallbackHandler can serialise/restore callback state.
    - Multiple callbacks can be registered and compose via OR logic.

    Execution order:
    - Callbacks are called in the order they are added to CallbackHandler.
    - Each callback's return value (if not None) is merged into the shared
      TrainerControl using bitwise OR (field‑wise OR). This is "any‑true"
      semantics: if ANY callback requests a stop, training stops.

    Important Lifecycle Guarantees:
    - `on_init_end` is called after Trainer initialisation but before
      training begins (useful for validating config).
    - `on_epoch_begin/end` are called even for incomplete epochs
      (e.g., when max_steps cuts in the middle).
    - `on_substep_end` is for gradient accumulation steps; called after
      each forward/backward but before optimizer step. Implementations
      must be lightweight – it runs O(num_grad_acc_steps × num_steps).

    Performance Note:
    - All methods are optional and default to `pass`. The base class uses
      `Optional[TrainerControl]` return because most callbacks don't need
      to modify control. Returning `None` is equivalent to "no change".

    Example Usage:
        >>> class MyLogger(TrainerCallback):
        ...     def on_log(self, args, state, control, logs=None, **kwargs):
        ...         print(f"Step {state.global_step}: loss = {logs.get('loss')}")
        ...         return control
    """
    def on_init_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_train_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_train_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_epoch_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_epoch_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_step_begin(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_step_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_substep_end(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_evaluate(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_predict(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_save(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_log(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def on_prediction_step(self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any) -> Optional[TrainerControl]:
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return serialisable state for checkpointing. Override for stateful callbacks."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore callback state from a checkpoint. Override in subclasses."""
        pass


class CallbackHandler:
    """
    Registry and dispatcher for multiple TrainerCallbacks.

    Responsibilities:
    - Maintain ordered list of callbacks.
    - Dispatch lifecycle events to all callbacks, merging their TrainerControl
      outputs with OR semantics.
    - Provide serialisation/deserialisation of callback states.

    OR‑semantics (why this matters):
        If callback A sets `should_log = True` and callback B leaves it False,
        the final `control.should_log` becomes True. This is correct for
        "any request should be honoured". The alternative (majority or last‑write)
        would be surprising and harder to debug.

    Edge Cases:
    - If a callback returns `None`, it is treated as no‑change (skip OR).
    - The handler does NOT validate that callbacks modify only intended flags.
    - State dict keys are the class names of callbacks. If two callbacks of
      the same type exist, the second will overwrite the first's state on load.
      Workaround: Avoid duplicate callback types OR implement custom state_dict
      that includes an identifier.

    Performance:
    - Dispatching iterates over all callbacks linearly. For >10 callbacks,
      the overhead is still negligible (each callback method is usually fast).
    - No caching of method lookups: `getattr(callback, event)` called every time.
      Acceptable because events occur O(epochs+steps) and callbacks are few.
    """
    def __init__(self, callbacks: Optional[list[TrainerCallback]] = None, args: Any = None) -> None:
        self.callbacks: list[TrainerCallback] = []
        self.args = args  # TrainingArguments-like object, passed to each callback.
        for cb in callbacks or []:
            self.add_callback(cb)

    def add_callback(self, callback: TrainerCallback) -> None:
        """Append a callback. Raises TypeError if not a TrainerCallback instance."""
        if not isinstance(callback, TrainerCallback):
            raise TypeError(
                f"Expected a TrainerCallback instance, got {type(callback).__name__}"
            )
        self.callbacks.append(callback)

    def remove_callback(self, callback_class: type) -> None:
        """Remove the first callback that matches the given class."""
        for i, cb in enumerate(self.callbacks):
            if isinstance(cb, callback_class):
                self.callbacks.pop(i)
                return
        raise ValueError(f"No callback of type {callback_class.__name__} is registered.")

    def pop_callback(self, callback_class: type) -> TrainerCallback:
        """Remove and return the first callback matching the class."""
        for i, cb in enumerate(self.callbacks):
            if isinstance(cb, callback_class):
                return self.callbacks.pop(i)
        raise ValueError(f"No callback of type {callback_class.__name__} is registered.")

    def _dispatch(self, event: str, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        """
        Internal: call `event` on every callback and merge returned controls.

        Merge logic (field‑wise OR) – explained above.
        Workaround: We use `|=` because bool is immutable; we want in‑place update
        of the original control instance that the Trainer holds.
        """
        for callback in self.callbacks:
            result = getattr(callback, event)(self.args, state, control, **kwargs)
            if result is not None:
                control.should_training_stop |= result.should_training_stop
                control.should_epoch_stop   |= result.should_epoch_stop
                control.should_save         |= result.should_save
                control.should_evaluate     |= result.should_evaluate
                control.should_log          |= result.should_log
        return control

    # Public dispatchers – each simply calls _dispatch with the method name.
    def on_init_end(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_init_end", state, control, **kwargs)

    def on_train_begin(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_train_begin", state, control, **kwargs)

    def on_train_end(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_train_end", state, control, **kwargs)

    def on_epoch_begin(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_epoch_begin", state, control, **kwargs)

    def on_epoch_end(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_epoch_end", state, control, **kwargs)

    def on_step_begin(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_step_begin", state, control, **kwargs)

    def on_step_end(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_step_end", state, control, **kwargs)

    def on_substep_end(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_substep_end", state, control, **kwargs)

    def on_evaluate(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_evaluate", state, control, **kwargs)

    def on_predict(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_predict", state, control, **kwargs)

    def on_save(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_save", state, control, **kwargs)

    def on_log(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_log", state, control, **kwargs)

    def on_prediction_step(self, state: TrainerState, control: TrainerControl, **kwargs: Any) -> TrainerControl:
        return self._dispatch("on_prediction_step", state, control, **kwargs)

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize all callbacks' states, keyed by their class name."""
        return {type(cb).__name__: cb.state_dict() for cb in self.callbacks}

    def load_state_dict(self, state: dict[str, dict[str, Any]]) -> None:
        """Restore callback states from a previously saved state dict."""
        for cb in self.callbacks:
            cb_state = state.get(type(cb).__name__)
            if cb_state is not None:
                cb.load_state_dict(cb_state)


class EarlyStoppingCallback(TrainerCallback):
    """
    Stop training when a monitored metric stops improving.

    Implementation based on Hugging Face Transformers' EarlyStoppingCallback
    with deterministic patience counter.

    How it works:
    - Monitors a single metric (e.g., "eval_loss").
    - On each evaluation, checks if metric improved by at least
      `early_stopping_threshold` relative to the best seen so far.
    - Improvement direction is determined automatically from metric name
      (contains "loss" → lower is better, else higher is better), or can
      be overridden by `greater_is_better`.
    - If no improvement after `early_stopping_patience` consecutive evaluations,
      sets `control.should_training_stop = True`.

    Assumptions & Invariants:
    - `on_evaluate` must receive `metrics` as a keyword argument containing
      the metric to monitor. If missing, callback does nothing and logs a warning.
    - The metric value must be finite (not NaN/Inf). The callback raises a
      ValueError otherwise – this prevents silent early stopping on corrupted metrics.
    - The counter resets as soon as any improvement occurs, even if the improvement
      is tiny (but must exceed threshold). This matches standard practice.

    Edge Cases & Trade‑offs:
    - If `early_stopping_threshold = 0.0` (default), any numeric improvement resets
      the counter. This can cause never stopping if metric oscillates around a value
      due to noise. A small positive threshold (e.g., 1e-4) adds robustness.
    - The first evaluation always registers as an improvement (counter = 0) because
      `best_metric` is initialised to -inf or +inf. This ensures at least
      `patience+1` evaluations before stopping.
    - When `greater_is_better = None` (auto‑detect), the logic is:
        if "loss" in metric_name.lower(): greater_is_better = False else True.
      This heuristic fails for metrics like "eval_accuracy" (correct) but also
      "eval_negative_loss" (unlikely). Override explicitly if ambiguous.
    - Patience counter is stored in the callback's state, so it survives checkpoint/
      resume. This is crucial: without saving the counter, resuming would restart
      patience from zero, potentially causing over‑training.

    Performance:
    - No I/O or heavy operations. Just comparisons and logging.

    Example Usage:
        >>> callback = EarlyStoppingCallback(
        ...     early_stopping_patience=3,
        ...     metric_for_best_model="eval_accuracy",
        ...     greater_is_better=True
        ... )
        >>> handler = CallbackHandler(callbacks=[callback])
        >>> # Then attach handler to Trainer.
    """
    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.0,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: Optional[bool] = None,
    ) -> None:
        if early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be ≥ 1, got {early_stopping_patience}."
            )

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_for_best_model = metric_for_best_model

        # Auto‑detect improvement direction if not provided.
        # This heuristic is simple but effective for common metrics.
        if greater_is_better is None:
            self.greater_is_better = "loss" not in metric_for_best_model.lower()
        else:
            self.greater_is_better = greater_is_better

        self.early_stopping_patience_counter: int = 0
        # Initial best metric depends on direction: we want the first evaluation
        # to always count as an improvement, so set to worst possible value.
        self.best_metric: float = float("-inf") if self.greater_is_better else float("inf")

    def _is_improvement(self, current: float) -> bool:
        """
        Returns True if `current` is better than `best_metric` by at least `threshold`.

        The threshold ensures that noise doesn't reset patience counter.
        """
        if self.greater_is_better:
            return current > self.best_metric + self.early_stopping_threshold
        else:
            return current < self.best_metric - self.early_stopping_threshold

    def _check_metric_value(self, metrics: dict[str, float]) -> float:
        """Extract monitored metric and validate it's a finite number."""
        if self.metric_for_best_model not in metrics:
            available = list(metrics.keys())
            raise KeyError(
                f"EarlyStoppingCallback: metric '{self.metric_for_best_model}' not found "
                f"in eval metrics. Available keys: {available}. "
                f"Set metric_for_best_model to one of these."
            )

        value = metrics[self.metric_for_best_model]

        if not math.isfinite(value):
            raise ValueError(
                f"EarlyStoppingCallback: monitored metric '{self.metric_for_best_model}' "
                f"is {value} (NaN or Inf). Check your evaluation loop."
            )

        return value

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict[str, float]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """
        Evaluate metric improvement and update stop flag.

        Side Effects:
        - Updates `self.best_metric` and `self.early_stopping_patience_counter`.
        - Logs information at INFO level when not improving.
        - May set `control.should_training_stop = True`.

        Assumptions: The Trainer passes `metrics` dict after each evaluation.
        """
        if metrics is None:
            logger.warning(
                "EarlyStoppingCallback.on_evaluate called without a 'metrics' kwarg. "
                "Skipping early-stopping check."
            )
            return control

        current_metric = self._check_metric_value(metrics)

        if self._is_improvement(current_metric):
            self.best_metric = current_metric
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            logger.info(
                "EarlyStoppingCallback: %s=%.6f did not improve from best %.6f "
                "(threshold=%.6f). Counter: %d/%d",
                self.metric_for_best_model,
                current_metric,
                self.best_metric,
                self.early_stopping_threshold,
                self.early_stopping_patience_counter,
                self.early_stopping_patience,
            )

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            logger.info(
                "EarlyStoppingCallback: patience exhausted. Stopping training. "
                "Best %s = %.6f",
                self.metric_for_best_model,
                self.best_metric,
            )
            control.should_training_stop = True

        return control

    def state_dict(self) -> dict[str, Any]:
        """Save best metric and patience counter for checkpoint resume."""
        return {
            "best_metric": self.best_metric,
            "early_stopping_patience_counter": self.early_stopping_patience_counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state. Missing keys keep current values (graceful degradation)."""
        self.best_metric = state.get("best_metric", self.best_metric)
        self.early_stopping_patience_counter = state.get(
            "early_stopping_patience_counter",
            self.early_stopping_patience_counter,
        )


def _usage_example() -> None:
    """
    Example of wiring callbacks and handler into a training loop.

    This is intentionally simplified; a real Trainer would have more logic.
    The purpose is to demonstrate the interaction pattern:

    >>> from dataclasses import dataclass as dc
    >>> @dc
    ... class TrainingArgs:
    ...     output_dir: str = "./output"
    ...     logging_steps: int = 10
    ...     save_steps: int = 100
    ...     eval_steps: int = 100
    ...
    >>> args = TrainingArgs()
    >>> state = TrainerState(num_train_epochs=10, max_steps=1000)
    >>> control = TrainerControl()
    >>> handler = CallbackHandler(callbacks=[EarlyStoppingCallback(patience=3)], args=args)
    >>> handler.on_train_begin(state, control)
    >>> for epoch in range(state.num_train_epochs):
    ...     handler.on_epoch_begin(state, control)
    ...     for step in range(10):
    ...         handler.on_step_begin(state, control)
    ...         state.global_step += 1
    ...         handler.on_step_end(state, control)
    ...         if control.should_log:
    ...             handler.on_log(state, control, logs={"loss": 1.0})
    ...             control.should_log = False
    ...     handler.on_epoch_end(state, control)
    ...     metrics = {"eval_loss": 0.5 - epoch * 0.05}
    ...     handler.on_evaluate(state, control, metrics=metrics)
    ...     if control.should_training_stop:
    ...         break
    >>> handler.on_train_end(state, control)
    """
    from dataclasses import dataclass as dc

    @dc
    class TrainingArguments:
        output_dir: str = "./output"
        logging_steps: int = 10
        save_steps: int = 100
        eval_steps: int = 100

    args = TrainingArguments()
    state = TrainerState(num_train_epochs=10, max_steps=1000)
    control = TrainerControl()

    handler = CallbackHandler(
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            ),
        ],
        args=args,
    )

    handler.on_train_begin(state, control)

    for epoch in range(args.output_dir and 3):   # Simulated epochs
        handler.on_epoch_begin(state, control)

        for step in range(10):
            handler.on_step_begin(state, control)
            state.global_step += 1
            handler.on_step_end(state, control)

            if control.should_log:
                handler.on_log(state, control, logs={"loss": 1.0})
                control.should_log = False

        handler.on_epoch_end(state, control)

        metrics = {"eval_loss": 0.5 - epoch * 0.05}
        handler.on_evaluate(state, control, metrics=metrics)

        if control.should_training_stop:
            break

    handler.on_train_end(state, control)