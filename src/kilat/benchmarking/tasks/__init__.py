"""Synthetic benchmark tasks for Kilat model certification."""
from .induction_heads import InductionAnalyticalOracle, InductionHeadsConfig
from .mqar import MQARAnalyticalOracle, MQARConfig
from .selective_copying import SelectiveCopyingConfig, SelectiveCopyingOracle

__all__ = [
    "MQARConfig",
    "MQARAnalyticalOracle",
    "InductionHeadsConfig",
    "InductionAnalyticalOracle",
    "SelectiveCopyingConfig",
    "SelectiveCopyingOracle",
]
