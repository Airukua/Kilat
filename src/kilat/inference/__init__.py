"""Canonical text generation and Hugging Face checkpoint export package."""

from typing import TYPE_CHECKING, Any

from kilat.inference.beam_search import (
    BeamHypotheses,
    BeamSearchScorer,
    beam_search_generate,
)
from kilat.inference.generation_config import GenerationConfig
from kilat.inference.generation_mixin import GenerationMixin
from kilat.inference.generator import (
    TextGenerator,
    batch_generate,
    quick_generate,
    stream_generate,
)
from kilat.inference.logit_processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    get_logits_processor,
    get_logits_warper,
)
from kilat.inference.sampler import (
    AdaptiveSampler,
    ContrastiveSampler,
    GreedySampler,
    MirostatSampler,
    MultinomialSampler,
    ReturnValue,
    Sampler,
    TopKSampler,
    TopKTopPSampler,
    TopPSampler,
    TypicalSampler,
    get_sampler,
)
from kilat.inference.stopping_criteria import (
    CompositeCriteria,
    CustomCriteria,
    EndStringCriteria,
    EosTokenCriteria,
    IterationLimitCriteria,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MinLengthCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    TimeoutCriteria,
    get_stopping_criteria,
)

if TYPE_CHECKING:
    from kilat.inference.export_hf import (
        ConfigNotFoundError,
        ConversionError,
        ValidationError,
        WeightFileNotFoundError,
        WeightLoadingError,
        convert_checkpoint_to_hf,
    )

_EXPORT_HF_NAMES = {
    "ConversionError",
    "ConfigNotFoundError",
    "WeightFileNotFoundError",
    "WeightLoadingError",
    "ValidationError",
    "convert_checkpoint_to_hf",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_HF_NAMES:
        from kilat.inference import export_hf

        return getattr(export_hf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TextGenerator",
    "batch_generate",
    "quick_generate",
    "stream_generate",
    "GenerationConfig",
    "GenerationMixin",
    "BeamHypotheses",
    "BeamSearchScorer",
    "beam_search_generate",
    "LogitsProcessor",
    "LogitsProcessorList",
    "ForcedBOSTokenLogitsProcessor",
    "ForcedEOSTokenLogitsProcessor",
    "MinLengthLogitsProcessor",
    "NoRepeatNGramLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "SuppressTokensLogitsProcessor",
    "TemperatureLogitsWarper",
    "TopKLogitsWarper",
    "TopPLogitsWarper",
    "get_logits_processor",
    "get_logits_warper",
    "Sampler",
    "ReturnValue",
    "GreedySampler",
    "MultinomialSampler",
    "TopKSampler",
    "TopPSampler",
    "TopKTopPSampler",
    "TypicalSampler",
    "ContrastiveSampler",
    "MirostatSampler",
    "AdaptiveSampler",
    "get_sampler",
    "StoppingCriteria",
    "StoppingCriteriaList",
    "MaxLengthCriteria",
    "MaxNewTokensCriteria",
    "MinLengthCriteria",
    "EosTokenCriteria",
    "EndStringCriteria",
    "TimeoutCriteria",
    "IterationLimitCriteria",
    "CustomCriteria",
    "CompositeCriteria",
    "get_stopping_criteria",
    "ConversionError",
    "ConfigNotFoundError",
    "WeightFileNotFoundError",
    "WeightLoadingError",
    "ValidationError",
    "convert_checkpoint_to_hf",
]
