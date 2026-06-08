from __future__ import annotations
from typing import Any, Optional


class TrainerCallback:
    """
    Base class for trainer lifecycle hooks.

    WHY: Callbacks enable non‑invasive extensions (early stopping, logging,
    checkpointing) without bloating the trainer core. Each method is a hook
    that subclasses override.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called once before the training loop starts."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called once after the training loop finishes (including early stop)."""
        pass

    def on_evaluate_end(self, trainer: Any, eval_loss: float, eval_ppl: float) -> bool:
        """
        Called after each evaluation step.

        Returns: True if training should stop early (e.g., early stopping triggered).
        """
        return False

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state for checkpointing (e.g., best_loss counter)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore callback state from a checkpoint."""
        pass


class CallbackHandler:
    """
    Dispatches lifecycle events to all registered callbacks.

    WHY: Centralises iteration over callbacks. The trainer only talks to the
    handler, not individual callbacks. This also handles state aggregation.
    """

    def __init__(self, callbacks: Optional[list[TrainerCallback]] = None) -> None:
        self.callbacks = list(callbacks or [])

    def add_callback(self, callback: TrainerCallback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_evaluate_end(self, trainer: Any, eval_loss: float, eval_ppl: float) -> bool:
        """Return True if any callback requests early stopping."""
        should_stop = False
        for callback in self.callbacks:
            # Use OR: once stop is requested, keep it True even if later callbacks return False.
            should_stop = callback.on_evaluate_end(trainer, eval_loss, eval_ppl) or should_stop
        return should_stop

    def state_dict(self) -> dict[str, dict[str, Any]]:
        """Aggregate state dicts using callback class names as keys."""
        return {
            callback.__class__.__name__: callback.state_dict()
            for callback in self.callbacks
        }

    def load_state_dict(self, state: dict[str, dict[str, Any]]) -> None:
        """Restore each callback from its respective sub‑dict."""
        for callback in self.callbacks:
            callback_state = state.get(callback.__class__.__name__)
            if callback_state is not None:
                callback.load_state_dict(callback_state)


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping based on evaluation loss.

    Stops training when the evaluation loss does not improve for `patience`
    consecutive evaluations. Improvement is defined as `current_loss < best_loss - threshold`.

    WHY: Prevents overfitting and wasted compute when the model plateaus.
    The `threshold` parameter avoids stopping on micro‑fluctuations.

    Edge cases:
    - If eval_loss is NaN, it will never improve and will eventually stop.
    - The callback maintains internal state (best_loss, fail_counter) that
      can be checkpointed and resumed.
    """

    def __init__(self, patience: int, threshold: float = 0.0) -> None:
        """
        Args:
            patience: Number of evaluations with no improvement before stopping.
            threshold: Minimum reduction in loss to count as improvement.
                       Set >0 to be less sensitive to small changes.
        """
        self.patience = patience
        self.threshold = threshold
        self.best_loss: float = float("inf")
        self.fail_counter: int = 0
        self.should_stop: bool = False

    def check(self, current_eval_loss: float) -> bool:
        """
        Update internal counters and return whether training should stop.

        Decision logic:
        - If current_loss is better than best_loss (by at least threshold),
          reset fail counter and update best_loss.
        - Otherwise, increment fail counter. If counter reaches patience,
          set should_stop = True.
        """
        if current_eval_loss < self.best_loss - self.threshold:
            self.best_loss = current_eval_loss
            self.fail_counter = 0
        else:
            self.fail_counter += 1
            # Provide immediate feedback so the user knows progress stalled.
            print(
                f"\n[EarlyStopping] Eval loss {current_eval_loss:.4f} did not improve "
                f"over best {self.best_loss:.4f}. "
                f"Counter: {self.fail_counter}/{self.patience}"
            )
            if self.fail_counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def on_evaluate_end(self, trainer: Any, eval_loss: float, eval_ppl: float) -> bool:
        """Hook called by trainer after evaluation."""
        return self.check(eval_loss)

    def state_dict(self) -> dict[str, Any]:
        """Save best_loss, fail_counter, and should_stop for checkpoint resume."""
        return {
            "best_loss": self.best_loss,
            "fail_counter": self.fail_counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore early stopping state from a checkpoint."""
        self.best_loss = state.get("best_loss", self.best_loss)
        self.fail_counter = state.get("fail_counter", self.fail_counter)
        self.should_stop = state.get("should_stop", self.should_stop)