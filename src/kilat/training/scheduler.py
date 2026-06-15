from __future__ import annotations
import math
import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional
from enum import Enum
import torch
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# SchedulerType enum – defines all built‑in scheduler identifiers.
# This is required by the factory function `get_scheduler`.
# ----------------------------------------------------------------------------
class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    WSDLR = "wsdlr"
    REX = "rex"


# ----------------------------------------------------------------------------
# Abstract Base Class
# ----------------------------------------------------------------------------
class LRScheduler(ABC):
    """
    Abstract base class for all learning rate schedulers.

    WHY: Enforces a consistent interface so the factory function (``get_scheduler``)
    and the trainer can treat every scheduler identically. The only thing a
    subclass must provide is ``get_lr_lambda()``.

    The base class handles:
    - Wrapping the lambda in ``torch.optim.lr_scheduler.LambdaLR``.
    - Exposing ``step()``, ``state_dict()``, and ``load_state_dict()`` as thin
      pass‑throughs to the underlying PyTorch scheduler.
    - Logging the scheduler configuration at construction time.

    Subclassing contract
    --------------------
    Override ``get_lr_lambda()`` to return a ``Callable[[int], float]``.
    The callable receives a global step (int, 0-indexed) and must return a
    multiplier in ``[0, 1]`` (or slightly above 1 during warmup if desired,
    but standard practice is [0, 1]).

    Do NOT override ``__init__`` without calling ``super().__init__()``.

    Example Usage
    -------------
        >>> class MyScheduler(LRScheduler):
        ...     def get_lr_lambda(self):
        ...         return lambda step: 1.0 / (1.0 + 0.01 * step)
        >>> sched = MyScheduler(optimizer, num_warmup_steps=0, num_training_steps=1000)
        >>> for step in range(1000):
        ...     train_step()
        ...     sched.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer whose LR groups will be scheduled.
        num_warmup_steps : int
            Number of linear warmup steps from 0 → peak LR.
        num_training_steps : int
            Total number of optimizer steps (including warmup).
        last_epoch : int
            The index of the last epoch. Default: -1 (fresh start).
            Pass a non‑negative value to resume from a checkpoint.

        Raises
        ------
        ValueError
            If warmup steps exceed total steps, or if arguments are invalid.
        """
        if num_warmup_steps < 0:
            raise ValueError(
                f"num_warmup_steps must be >= 0, got {num_warmup_steps}."
            )
        if num_training_steps < 1:
            raise ValueError(
                f"num_training_steps must be >= 1, got {num_training_steps}."
            )
        if num_warmup_steps > num_training_steps:
            raise ValueError(
                f"num_warmup_steps ({num_warmup_steps}) must not exceed "
                f"num_training_steps ({num_training_steps})."
            )

        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self.get_lr_lambda(),
            last_epoch=last_epoch,
        )

        logger.info(
            "%s: warmup_steps=%d, total_steps=%d",
            self.__class__.__name__,
            num_warmup_steps,
            num_training_steps,
        )

    @abstractmethod
    def get_lr_lambda(self) -> Callable[[int], float]:
        """
        Return a callable that maps a global step to an LR multiplier.

        The returned function must:
        - Accept a single ``int`` argument (current global step, 0‑indexed).
        - Return a ``float`` multiplier, typically in ``[0.0, 1.0]``.
        - Be a pure function with no side effects (PyTorch calls it internally).

        Example implementation
        ----------------------
        >>> def get_lr_lambda(self):
        ...     def lr_lambda(current_step: int) -> float:
        ...         if current_step < self.num_warmup_steps:
        ...             return current_step / max(1, self.num_warmup_steps)
        ...         return 1.0  # constant after warmup
        ...     return lr_lambda
        """

    # ------------------------------------------------------------------
    # Pass‑throughs to the underlying LambdaLR
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the scheduler by one optimizer step."""
        self._scheduler.step()

    def state_dict(self) -> dict:
        """Return the scheduler state for checkpointing."""
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        self._scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        """Return the LR computed at the last ``step()`` call."""
        return self._scheduler.get_last_lr()

    def get_lr(self) -> list[float]:
        """Return the current LR for all optimizer param groups."""
        return self._scheduler.get_lr()

    # ------------------------------------------------------------------
    # Shared warmup helper — reused by most subclasses
    # ------------------------------------------------------------------

    def _warmup_factor(self, current_step: int) -> Optional[float]:
        """
        Return the warmup multiplier if still in warmup phase, else None.

        This helper factorises the common linear warmup logic.
        It is called from the `get_lr_lambda` implementations of subclasses.

        Returns
        -------
        Optional[float]
            - If `current_step < self.num_warmup_steps`: linear multiplier
              in `[0, 1]`.
            - Otherwise: `None` (warmup finished).

        Usage pattern in subclasses::

            def get_lr_lambda(self):
                def lr_lambda(step):
                    wf = self._warmup_factor(step)
                    if wf is not None:
                        return wf
                    # ... decay logic ...
                return lr_lambda
        """
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return None


# ---------------------------------------------------------------------------
# Concrete schedulers
# ---------------------------------------------------------------------------

class LinearScheduler(LRScheduler):
    """
    Linear warmup followed by linear decay to 0.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Decay   : 1 → 0  (linear, over remaining steps)

    Trade‑offs
    ----------
    + Very simple, no hyperparameters.
    - Linear decay may be too abrupt at the end, causing training instability
      for large models. Cosine decay is generally preferred for LLMs.

    Use case
    --------
    Baseline scheduler. Works well for fine‑tuning smaller models where
    aggressive early large LR is acceptable.

    Example Usage
    -------------
        >>> scheduler = LinearScheduler(optimizer, num_warmup_steps=100, num_training_steps=1000)
        >>> for step in range(1000):
        ...     train_step()
        ...     scheduler.step()
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            # Linear decay: maps [warmup_steps, total_steps] → [1, 0]
            remaining = self.num_training_steps - self.num_warmup_steps
            decay_steps = current_step - self.num_warmup_steps
            return max(0.0, float(remaining - decay_steps) / float(max(1, remaining)))

        return lr_lambda


class CosineScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing to 0.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Decay   : 1 → 0  (cosine half‑period, over remaining steps)

    WHY cosine: The cosine curve decays slowly at the start of the decay phase
    (allowing the model to remain near peak learning rate for longer), then
    decays faster in the middle, and slows again near zero (fine convergence).
    This empirically outperforms linear decay for transformer pre‑training.

    Reference: SGDR (Loshchilov & Hutter, 2017), single‑cycle variant.

    Assumptions
    -----------
    - The total number of steps (`num_training_steps`) is known exactly.
    - Warmup steps are a small fraction (1‑10%) of total steps.

    Use case
    --------
    Default scheduler for LLM pre‑training and most fine‑tuning. Used in
    GPT‑2, GPT‑3, LLaMA, Falcon, and the majority of modern LLMs.

    Example Usage
    -------------
        >>> scheduler = CosineScheduler(optimizer, num_warmup_steps=500, num_training_steps=10000)
        >>> for step in range(10000):
        ...     train_step()
        ...     scheduler.step()
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            decay_steps = self.num_training_steps - self.num_warmup_steps
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, decay_steps)
            )
            # 0.5 * (1 + cos(π * progress)) maps progress ∈ [0,1] → multiplier ∈ [1,0]
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_lambda


class CosineWithRestartsScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing with hard periodic restarts.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Decay   : Repeated cosine cycles of equal length. At the end of each
              cycle the LR is reset to 1 (peak), then decays again.

    Parameters
    ----------
    num_cycles : int
        Number of cosine cycles in the decay phase (default: 1 = standard cosine).
        Each cycle spans ``(total - warmup) / num_cycles`` steps.

    WHY restarts: Each restart acts as a learning rate spike that can help
    the optimizer escape sharp local minima. The exploration benefit is most
    pronounced with cycle lengths > 1 epoch.

    Reference: SGDR (Loshchilov & Hutter, 2017).

    Edge Cases
    ----------
    - If `num_cycles` is 1, behaviour is identical to `CosineScheduler`.
    - If `num_cycles` is large, cycles become short and the LR may restart
      before significant decay, effectively behaving like a high‑frequency
      cyclical schedule.

    Use case
    --------
    Continual pre‑training where you want multiple "bursts" of exploration,
    or ensemble‑by‑checkpoint: save a model at each restart trough and
    ensemble the predictions.

    Example Usage
    -------------
        >>> scheduler = CosineWithRestartsScheduler(
        ...     optimizer, num_warmup_steps=100, num_training_steps=10000, num_cycles=4
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ) -> None:
        self.num_cycles = num_cycles
        super().__init__(optimizer, num_warmup_steps, num_training_steps, last_epoch)

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            decay_steps = self.num_training_steps - self.num_warmup_steps
            # progress within the current cycle, in [0, 1)
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, decay_steps)
            )
            cycle_progress = math.fmod(progress * self.num_cycles, 1.0)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle_progress)))

        return lr_lambda


class CosineWithMinLRScheduler(LRScheduler):
    """
    Cosine annealing that decays to a non‑zero floor (``min_lr_ratio × peak_lr``).

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear)
    Decay   : 1 → ``min_lr_ratio``  (cosine), never going below the floor.

    Parameters
    ----------
    min_lr_ratio : float
        Floor as a fraction of the peak learning rate (default: 0.1 = 10%).
        Common values: 0.01–0.1.

    WHY a floor: Standard cosine drives LR to 0, which can cause the optimizer
    to effectively stop updating late in training. A small non‑zero floor keeps
    updates alive and is especially important for very long training runs or when
    the learning rate is resumed from a checkpoint at a new phase.

    Assumptions
    -----------
    - `min_lr_ratio` must be in `[0, 1)`. Zero is allowed (degenerates to standard cosine).
    - The peak LR is the optimizer's base LR (set before scheduler is attached).

    Use case
    --------
    Llama 2 / Llama 3 style training (10% floor). Long pre‑training runs where
    complete LR zeroing causes premature convergence.

    Example Usage
    -------------
        >>> scheduler = CosineWithMinLRScheduler(
        ...     optimizer, num_warmup_steps=500, num_training_steps=10000, min_lr_ratio=0.05
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if not 0.0 <= min_lr_ratio < 1.0:
            raise ValueError(
                f"min_lr_ratio must be in [0, 1), got {min_lr_ratio}."
            )
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, num_warmup_steps, num_training_steps, last_epoch)

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            decay_steps = self.num_training_steps - self.num_warmup_steps
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, decay_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Rescale cosine from [0,1] to [min_lr_ratio, 1]
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        return lr_lambda


class PolynomialScheduler(LRScheduler):
    """
    Linear warmup followed by polynomial (power‑law) decay.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear)
    Decay   : 1 → ``lr_end_ratio``  via (1 - progress)^power

    Parameters
    ----------
    power : float
        Exponent of the polynomial (default: 1.0 = linear decay).
        - power < 1: concave decay (fast early, slow late)
        - power = 1: linear decay
        - power > 1: convex decay (slow early, fast late)
        Common choice: power=2 or power=3 for quadratic/cubic decay.
    lr_end_ratio : float
        Floor as a fraction of peak LR (default: 0.0).

    WHY polynomial: More flexible than cosine. With power=1 it matches linear,
    and power>1 gives decay profiles that can be tuned for specific tasks.
    BERT uses polynomial decay with power=1 (effectively linear).

    Edge Cases
    ----------
    - If `lr_end_ratio = 0`, LR decays exactly to zero.
    - If `power = 0` is not allowed (would cause division by zero). The code
      raises `ValueError` for `power <= 0`.

    Use case
    --------
    BERT‑style fine‑tuning. Scenarios where you want explicit control over
    the curvature of the decay (not just cosine or linear).

    Example Usage
    -------------
        >>> scheduler = PolynomialScheduler(
        ...     optimizer, num_warmup_steps=100, num_training_steps=1000,
        ...     power=2.0, lr_end_ratio=0.01
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        power: float = 1.0,
        lr_end_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}.")
        if not 0.0 <= lr_end_ratio < 1.0:
            raise ValueError(f"lr_end_ratio must be in [0, 1), got {lr_end_ratio}.")
        self.power = power
        self.lr_end_ratio = lr_end_ratio
        super().__init__(optimizer, num_warmup_steps, num_training_steps, last_epoch)

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            if current_step >= self.num_training_steps:
                return self.lr_end_ratio
            decay_steps = self.num_training_steps - self.num_warmup_steps
            pct_remaining = 1.0 - float(current_step - self.num_warmup_steps) / float(
                max(1, decay_steps)
            )
            # Polynomial: pct_remaining^power rescaled to [lr_end_ratio, 1]
            decay = pct_remaining ** self.power
            return self.lr_end_ratio + (1.0 - self.lr_end_ratio) * decay

        return lr_lambda


class ConstantScheduler(LRScheduler):
    """
    Constant learning rate throughout training (no warmup, no decay).

    Schedule shape
    --------------
    All steps : 1.0

    WHY: Useful for debugging, quick experiments, or when the optimizer
    itself implements adaptive LR (e.g. Adafactor with ``scale_parameter=True``).

    Note: ``num_warmup_steps`` is accepted for API consistency but ignored.

    Use case
    --------
    Debugging the training loop. Adafactor‑based training where you want
    the optimizer's own adaptive mechanism to handle all LR scaling.

    Example Usage
    -------------
        >>> scheduler = ConstantScheduler(optimizer, num_warmup_steps=0, num_training_steps=1000)
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            return 1.0

        return lr_lambda


class ConstantWithWarmupScheduler(LRScheduler):
    """
    Linear warmup to peak LR, then constant for the rest of training.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Constant: 1.0    (for all remaining steps)

    WHY: Keeps the learning rate at its peak after warmup without any decay.
    Useful when external mechanisms (e.g. early stopping or a downstream
    evaluation‑based checkpoint selector) determine when to stop, rather than
    a scheduled decay.

    Use case
    --------
    Short fine‑tuning runs where the training budget is uncertain and you
    don't want the LR to decay to near‑zero before the run ends.
    RL fine‑tuning (RLHF/PPO) where you want stable LR throughout.

    Example Usage
    -------------
        >>> scheduler = ConstantWithWarmupScheduler(optimizer, num_warmup_steps=100, num_training_steps=1000)
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            return 1.0

        return lr_lambda


class InverseSqrtScheduler(LRScheduler):
    """
    Linear warmup followed by inverse‑square‑root decay: 1/√t.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Decay   : sqrt(warmup_steps) / sqrt(t)  for t > warmup_steps

    WHY 1/√t: The decay is slow but continuous, never reaching zero within
    any finite number of steps. The rate of decrease slows over time, meaning
    the model spends proportionally more time at moderate learning rates
    compared to cosine. This matches the theoretical optimal schedule for
    convex problems (AdaGrad analysis).

    The multiplier is normalised so that at `t = num_warmup_steps` the
    output equals 1.0 (seamless continuation from the warmup phase):
        multiplier(t) = sqrt(warmup_steps) / sqrt(t)
        multiplier(warmup_steps) = sqrt(warmup_steps)/sqrt(warmup_steps) = 1.0 ✓

    Reference: "Attention Is All You Need" (Vaswani et al., 2017);
               T5 (Raffel et al., 2020); Adafactor (Shazeer & Stern, 2018).

    Edge Cases
    ----------
    - If `num_warmup_steps = 0`, the warmup factor is never used and the
      formula `sqrt(1)/sqrt(t)` is used (starts from 1.0 at step 1).
    - The denominator `max(1, current_step)` avoids division by zero at step 0.

    Use case
    --------
    T5, mT5, and Adafactor‑based models. Training from scratch where you
    want a schedule that never zeros out the learning rate.

    Example Usage
    -------------
        >>> scheduler = InverseSqrtScheduler(optimizer, num_warmup_steps=500, num_training_steps=10000)
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        # Pre‑compute the warmup normalisation factor once.
        # If warmup_steps == 0, fall back to 1 to avoid sqrt(0).
        warmup_steps = max(1, self.num_warmup_steps)

        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            # Normalise so the multiplier equals 1.0 at the first post‑warmup step.
            return math.sqrt(warmup_steps) / math.sqrt(max(1, current_step))

        return lr_lambda


class WSDLRScheduler(LRScheduler):
    """
    Warmup‑Stable‑Decay (WSD) three‑phase schedule.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Stable  : 1.0    (constant, over the middle bulk of training)
    Decay   : 1 → 0  (linear, over ``num_decay_steps`` final steps)

    Parameters
    ----------
    num_decay_steps : int
        Number of final steps for the linear decay phase.
        If not set, defaults to ``num_warmup_steps`` (symmetric).

    WHY WSD: Proposed in MiniCPM (Hu et al., 2024), WSD decouples the
    stable training phase from the decay phase. The key insight is that
    the stable phase can be checkpointed and the model resumed from any
    point with a short decay phase applied, making it easy to produce
    models at multiple compute budgets from a single training run.

    The stable phase also enables continual pre‑training: a model trained
    with WSD can be extended cheaply — just continue the stable phase, then
    apply the decay phase again at the new budget.

    Reference: "MiniCPM: Scaling Inference‑Efficient Small Language Models"
               (Hu et al., 2024), https://arxiv.org/abs/2404.06395

    Assumptions
    -----------
    - The stable phase must have at least one step (i.e., `num_decay_steps < total_steps - warmup_steps`).
    - The decay phase uses linear decay; other decay shapes could be added but are not implemented.

    Use case
    --------
    LLM pre‑training with continual learning or multi‑budget evaluation.
    MiniCPM, CPM‑Bee, and follow‑up work in efficient LLM training.

    Example Usage
    -------------
        >>> scheduler = WSDLRScheduler(
        ...     optimizer, num_warmup_steps=100, num_training_steps=10000, num_decay_steps=500
        ... )
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_decay_steps: Optional[int] = None,
        last_epoch: int = -1,
    ) -> None:
        # Default decay steps = warmup steps (symmetric ramp‑up / ramp‑down).
        self.num_decay_steps = (
            num_decay_steps if num_decay_steps is not None else num_warmup_steps
        )
        if self.num_decay_steps >= num_training_steps - num_warmup_steps:
            raise ValueError(
                f"num_decay_steps ({self.num_decay_steps}) must be less than "
                f"(num_training_steps - num_warmup_steps) = "
                f"{num_training_steps - num_warmup_steps} so that the stable "
                f"phase has at least one step."
            )
        super().__init__(optimizer, num_warmup_steps, num_training_steps, last_epoch)

    def get_lr_lambda(self) -> Callable[[int], float]:
        stable_end = self.num_training_steps - self.num_decay_steps

        def lr_lambda(current_step: int) -> float:
            # Phase 1: warmup
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            # Phase 2: stable (constant at peak LR)
            if current_step < stable_end:
                return 1.0
            # Phase 3: linear decay → 0
            decay_progress = float(current_step - stable_end) / float(
                max(1, self.num_decay_steps)
            )
            return max(0.0, 1.0 - decay_progress)

        return lr_lambda


class REXScheduler(LRScheduler):
    """
    REX — Reflected Exponential scheduler.

    Schedule shape
    --------------
    Warmup  : 0 → 1  (linear, over ``num_warmup_steps``)
    Decay   : Exponential decay reflected so that the rate of decrease
              *accelerates* over time (unlike inverse‑sqrt which slows down).

    The decay formula (Ash & Adams, 2020):
        multiplier(t) = (1 - progress)^(1 / (1 - progress))   progress ∈ [0, 1)
    where `progress` is the fraction through the decay phase.

    At progress=0  : multiplier = 1.0   (peak LR, start of decay)
    At progress→1  : multiplier → 0.0   (LR collapses at the end)

    The function is convex and starts very slowly, then drops sharply near
    the end of training. This "saves" most of the budget for the stable LR
    phase and compresses the large‑LR benefit before a rapid final convergence.

    WHY REX: Empirically outperforms cosine on fixed compute budgets in
    vision and NLP. The intuition is that the model benefits from staying
    near peak LR longer, then making a sharp final descent.

    Edge case: At progress=1 (last step), `0^(1/0)` is undefined; clamped to 0.0.

    Reference: "REX: Revisiting Budgeted Training with an Improved Schedule"
               (Ash & Adams, 2020), https://arxiv.org/abs/2107.04197

    Use case
    --------
    Fixed‑budget training where you want to maximize the time spent at
    high learning rates and accept a sharp final convergence.

    Example Usage
    -------------
        >>> scheduler = REXScheduler(optimizer, num_warmup_steps=100, num_training_steps=10000)
    """

    def get_lr_lambda(self) -> Callable[[int], float]:
        def lr_lambda(current_step: int) -> float:
            wf = self._warmup_factor(current_step)
            if wf is not None:
                return wf
            decay_steps = self.num_training_steps - self.num_warmup_steps
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, decay_steps)
            )
            # Clamp to prevent domain errors at progress >= 1.
            if progress >= 1.0:
                return 0.0
            base = 1.0 - progress
            exponent = 1.0 / max(1e-9, base)
            return max(0.0, base ** exponent)

        return lr_lambda


# ---------------------------------------------------------------------------
# Factory function — single entry point for the trainer
# ---------------------------------------------------------------------------

_SCHEDULER_REGISTRY: dict[SchedulerType | str, type[LRScheduler]] = {
    SchedulerType.LINEAR: LinearScheduler,
    SchedulerType.COSINE: CosineScheduler,
    SchedulerType.COSINE_WITH_RESTARTS: CosineWithRestartsScheduler,
    SchedulerType.COSINE_WITH_MIN_LR: CosineWithMinLRScheduler,
    SchedulerType.POLYNOMIAL: PolynomialScheduler,
    SchedulerType.CONSTANT: ConstantScheduler,
    SchedulerType.CONSTANT_WITH_WARMUP: ConstantWithWarmupScheduler,
    SchedulerType.INVERSE_SQRT: InverseSqrtScheduler,
    SchedulerType.WSDLR: WSDLRScheduler,
    SchedulerType.REX: REXScheduler,
}


def get_scheduler(
    name: str | SchedulerType,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LRScheduler:
    """
    Factory function — instantiate any scheduler by name.

    This is the only function the trainer needs to call. Scheduler‑specific
    extra parameters (e.g. ``num_cycles``, ``min_lr_ratio``, ``power``,
    ``num_decay_steps``) are passed as ``**kwargs`` and forwarded to the
    appropriate class constructor.

    Parameters
    ----------
    name : str | SchedulerType
        Scheduler identifier. Must be one of the ``SchedulerType`` enum values.
        Accepts both the enum (``SchedulerType.COSINE``) and the string
        (``"cosine"``).
    optimizer : torch.optim.Optimizer
        Optimizer to attach the scheduler to.
    num_warmup_steps : int
        Number of warmup steps (0 = no warmup).
    num_training_steps : int
        Total number of optimizer steps.
    **kwargs
        Extra arguments forwarded to the scheduler constructor:
        - CosineWithRestartsScheduler : ``num_cycles`` (int, default 1)
        - CosineWithMinLRScheduler    : ``min_lr_ratio`` (float, default 0.1)
        - PolynomialScheduler         : ``power`` (float, default 1.0),
                                        ``lr_end_ratio`` (float, default 0.0)
        - WSDLRScheduler              : ``num_decay_steps`` (int, optional)

    Returns
    -------
    LRScheduler
        Instantiated scheduler, ready to call ``.step()`` after each
        optimizer step.

    Raises
    ------
    ValueError
        If ``name`` is not a valid scheduler identifier.

    External Dependencies
    ---------------------
    This function relies on the global registry `_SCHEDULER_REGISTRY`.
    Custom schedulers can be added via `register_scheduler()`.

    Example Usage
    -------------
        >>> # Basic cosine scheduler
        >>> scheduler = get_scheduler("cosine", optimizer, 100, 1000)
        >>> # Cosine with min LR
        >>> scheduler = get_scheduler(
        ...     SchedulerType.COSINE_WITH_MIN_LR, optimizer, 100, 1000,
        ...     min_lr_ratio=0.05,
        ... )
        >>> # WSD scheduler with custom decay steps
        >>> scheduler = get_scheduler(
        ...     "wsdlr", optimizer, num_warmup_steps=500, num_training_steps=10000,
        ...     num_decay_steps=1000
        ... )
    """
    if name in _SCHEDULER_REGISTRY:
        scheduler_cls = _SCHEDULER_REGISTRY[name]
    else:
        try:
            # Fall back to built-in enum names when the raw string key is not
            # registered. This keeps custom string schedulers accessible while
            # preserving support for SchedulerType values and built-ins.
            scheduler_type = SchedulerType(name)
        except ValueError:
            valid = [s.value for s in SchedulerType] + [
                key for key in _SCHEDULER_REGISTRY if isinstance(key, str)
            ]
            raise ValueError(
                f"Unknown scheduler '{name}'. Valid options: {valid}"
            ) from None

        scheduler_cls = _SCHEDULER_REGISTRY[scheduler_type]

    return scheduler_cls(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **kwargs,
    )


def register_scheduler(name: str, scheduler_cls: type[LRScheduler]) -> None:
    """
    Register a custom scheduler so it is accessible via ``get_scheduler()``.

    This allows external code to extend the scheduler system without modifying
    this file, following the open/closed principle.

    Parameters
    ----------
    name : str
        Unique string identifier for the new scheduler.
    scheduler_cls : type[LRScheduler]
        Class that inherits from ``LRScheduler`` and implements
        ``get_lr_lambda()``.

    Raises
    ------
    TypeError
        If ``scheduler_cls`` does not subclass ``LRScheduler``.
    ValueError
        If ``name`` is already registered (either as a built‑in or custom).

    Side Effects
    ------------
    Modifies the global registry `_SCHEDULER_REGISTRY`. After registration,
    `get_scheduler(name, ...)` will work.

    Example Usage
    -------------
        >>> class MyCustomScheduler(LRScheduler):
        ...     def get_lr_lambda(self):
        ...         return lambda step: 1.0 / (1.0 + 0.01 * step)
        ...
        >>> register_scheduler("my_custom", MyCustomScheduler)
        >>> scheduler = get_scheduler("my_custom", optimizer, 0, 1000)
    """
    if not (isinstance(scheduler_cls, type) and issubclass(scheduler_cls, LRScheduler)):
        raise TypeError(
            f"scheduler_cls must be a subclass of LRScheduler, "
            f"got {scheduler_cls}."
        )
    if name in {s.value for s in SchedulerType}:
        raise ValueError(
            f"'{name}' is already a built‑in scheduler type. "
            f"Choose a different name for your custom scheduler."
        )
    if name in _SCHEDULER_REGISTRY:
        raise ValueError(f"A scheduler named '{name}' is already registered.")
    # Store under the raw string key (not a SchedulerType enum member).
    _SCHEDULER_REGISTRY[name] = scheduler_cls
    logger.info("Registered custom scheduler: '%s' → %s", name, scheduler_cls.__name__)
