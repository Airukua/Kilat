from __future__ import annotations

class EarlyStoppingCallback:
    """
    Tracks evaluation loss and signals when training should stop based on
    plateau detection.

    Design Rationale
    ----------------
    Early stopping prevents overfitting and saves compute by terminating
    training when the model stops improving on held-out data. This implements
    the simplest and most widely-used variant: patience-based stopping with an
    absolute improvement threshold.

    The approach is intentionally simple (no moving averages, no statistical
    tests) because:
    1. It's deterministic and interpretable — easy to debug and explain.
    2. The threshold provides a natural knob: set to 0.0 for "any improvement
       resets patience", or to a small positive value (e.g., 1e-4) to require
       meaningful progress and avoid stopping due to noise.
    3. More complex criteria (e.g., requiring improvement over a moving
       average) rarely change the stopping decision in practice but add
       configuration burden.

    Statefulness
    -----------
    This object is stateful by design: it accumulates best_loss and fail_counter
    across multiple evaluations. This state MUST be included in checkpoint
    serialization to correctly resume early stopping behavior after preemption.
    Without state restoration, a restarted run would reset fail_counter to 0,
    effectively granting a fresh patience budget and potentially training far
    longer than intended.

    Best Loss Tracking
    ------------------
    self.best_loss is initialized to float("inf") so that the first evaluation
    ALWAYS counts as an improvement, regardless of the actual loss value. This
    ensures fail_counter starts at 0 and the model gets a full patience budget
    from the first evaluation onward. If initialized to 0, a poor first eval
    might immediately consume a patience strike.

    Parameters
    ----------
    patience : int
        Number of consecutive evaluations without sufficient improvement
        tolerated before stopping. Typical values: 3-10 for small datasets,
        2-5 for large-scale training where evaluations are expensive.
    threshold : float
        Minimum absolute decrease in loss to qualify as improvement.
        - 0.0 means any decrease resets patience (maximally sensitive).
        - Small positive values (e.g., 1e-4) filter out noise and require
          meaningful progress. This is especially important for large eval
          sets where loss variance is low and tiny fluctuations shouldn't
          reset the counter.
        - Negative values would be pathological (any increase counts as
          improvement) and are not explicitly guarded against.
    """

    def __init__(self, patience: int, threshold: float = 0.0) -> None:
        self.patience = patience
        self.threshold = threshold
        self.best_loss: float = float("inf")
        self.fail_counter: int = 0
        self.should_stop: bool = False

    def check(self, current_eval_loss: float) -> bool:
        """
        Evaluate current loss against best recorded loss and update state.

        Improvement criterion:
        current_eval_loss < best_loss - threshold

        Why strict less-than with a threshold?
        - Strict inequality (< not <=): Prevents identical loss values (within
          floating-point precision) from being considered improvements. This
          is important when the model has converged and loss oscillates at
          the precision limit.
        - Threshold is subtracted from best_loss: This means a loss of exactly
          best_loss - threshold counts as improvement. The threshold creates a
          "margin of significance" — only improvements larger than this margin
          reset the patience counter.

        Side Effects
        ------------
        - Updates self.best_loss when improvement is detected
        - Increments self.fail_counter when no improvement
        - Sets self.should_stop to True when patience is exhausted
        - Prints status message for non-improving evaluations (useful for
          monitoring training progress)

        Parameters
        ----------
        current_eval_loss : float
            Average evaluation loss from the most recent evaluation run.
            Expected to be positive and finite.

        Returns
        -------
        bool
            True if training should stop (patience exhausted), False otherwise.
            Once True, repeated calls continue returning True (sticky state).
        """
        if current_eval_loss < self.best_loss - self.threshold:
            # Improvement detected: reset counter and update best loss.
            # This resets the patience "clock" — the model gets fresh patience
            # from this point forward, which is the standard early stopping
            # behavior. Without reset, patience would be cumulative across all
            # evaluations, not consecutive non-improvements.
            self.best_loss = current_eval_loss
            self.fail_counter = 0
        else:
            # No significant improvement: increment failure counter.
            self.fail_counter += 1
            print(
                f"\n[EarlyStopping] Eval loss {current_eval_loss:.4f} did not improve "
                f"over best {self.best_loss:.4f}. "
                f"Counter: {self.fail_counter}/{self.patience}"
            )
            if self.fail_counter >= self.patience:
                self.should_stop = True

        return self.should_stop