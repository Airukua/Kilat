from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class BaseTeacher(ABC, nn.Module):
    """
    Interface for all teacher models in distillation.

    WHY: Distillation requires a frozen teacher that provides logits/soft labels.
    This abstraction allows swapping between different teacher sources (local
    Kilat checkpoints, HuggingFace models, custom APIs) without changing student code.

    Key design decisions:
    - Teacher is always frozen (no gradients, no training). We enforce this by
      overriding `train()` to be a no‑op and calling `_freeze()` in subclasses.
    - `forward()` must return logits of shape [B, T, vocab_size] (not probabilities).
    - `load()` is a factory method per subclass → each backend can have different
      loading arguments (e.g., device, dtype, trust_remote_code).
    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> "BaseTeacher":
        """Load teacher from any source (checkpoint, HF hub, API, etc.)."""
        ...

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return logits: Tensor [batch, seq_len, vocab_size]."""
        ...

    def _freeze(self):
        """Freeze all parameters and set to eval mode."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        """
        Override to prevent accidental training. Teacher stays in eval mode always.

        WHY: Some training loops call `model.train()` on all modules. This override
        ensures teacher never leaves eval mode, which would break distillation
        (e.g., dropout would introduce noise, batch norm would track statistics).
        """
        return self

    @property
    def vocab_size(self) -> int:
        """Return teacher's vocabulary size (must match student's for distillation loss)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            n = sum(p.numel() for p in self.parameters()) / 1e6
            return f"{self.__class__.__name__}({n:.1f}M params, frozen)"
        except Exception:
            return f"{self.__class__.__name__}(frozen)"


class KilatTeacher(BaseTeacher):
    """
    Teacher loaded from a Kilat training checkpoint (config.yaml + model.safetensors).

    WHY: Supports both safetensors and legacy .pt files. Uses the same
    `KilatTransformer` architecture as the student, ensuring identical tokenizer
    and embedding alignment. This is the fastest path for self‑distillation or
    progressive distillation from a larger Kilat model.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._freeze()

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> "KilatTeacher":
        """
        Load Kilat teacher from a checkpoint directory.

        Assumptions:
        - `config.yaml` exists and follows MainConfig format.
        - Weights are either `model.safetensors` (preferred) or `model.pt`.
        - The model architecture is exactly the same as the student's,
          so we can instantiate `KilatTransformer(main_cfg.model)` directly.
        """
        from arc.model import KilatTransformer
        from config import MainConfig
        from safetensors.torch import load_file

        checkpoint_dir = Path(checkpoint_dir)
        main_cfg = MainConfig.from_yaml(str(checkpoint_dir / "config.yaml"))
        model = KilatTransformer(main_cfg.model)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            # safetensors is faster and safer (no pickle)
            model.load_state_dict(load_file(str(safetensors_path), device=str(device)))
        else:
            # Fallback to legacy PyTorch checkpoint
            pt_path = checkpoint_dir / "model.pt"
            model.load_state_dict(torch.load(str(pt_path), map_location=device))

        return cls(model.to(device))

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        # KilatTransformer may return a tuple or a dict with .logits; we normalize.
        return out if isinstance(out, torch.Tensor) else out.logits

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size


class HuggingFaceTeacher(BaseTeacher):
    """
    Teacher from any HuggingFace causal LM (GPT‑2, LLaMA, Mistral, Qwen, etc.).

    WHY: Enables distillation from off‑the‑shelf large models without retraining.
    The only requirement is that the teacher's tokenizer vocabulary is compatible
    with the student's (or we handle mapping via a projection layer elsewhere).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._freeze()

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
    ) -> "HuggingFaceTeacher":
        """
        Load a HuggingFace AutoModelForCausalLM.

        Trade‑offs:
        - Using `AutoModelForCausalLM` is generic but may be slower than
          loading a specific class. We accept that for flexibility.
        - `torch_dtype` can be set to bfloat16 to save memory when teacher is huge.
        - `trust_remote_code=True` may be needed for custom architectures.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)

        return cls(model)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        # HF models return a CausalLMOutputWithPast, which has .logits attribute.
        return out.logits

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size


_REGISTRY: dict[str, type[BaseTeacher]] = {
    "kilat": KilatTeacher,
    "huggingface": HuggingFaceTeacher,
}

def load_teacher(backend: str, *args, **kwargs) -> BaseTeacher:
    """
    Single entry point to load any registered teacher.

    WHY: The factory pattern hides backend‑specific loading logic from the caller.
    The training script only needs to know the backend name and its arguments.

    Args:
        backend: "kilat", "huggingface", or any custom registered name.
        *args, **kwargs: forwarded to the teacher's `.load()` method.

    Example:
        teacher = load_teacher("kilat", "checkpoints/epoch-5", device="cuda")
        teacher = load_teacher("huggingface", "Qwen/Qwen2-1.5B", torch_dtype=torch.bfloat16)
    """
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[backend].load(*args, **kwargs)

def register_teacher(name: str, cls: type[BaseTeacher]):
    """
    Register a custom teacher class for use with `load_teacher`.

    WHY: Allows users to plug in custom teachers (e.g., from a remote API,
    a distilled ensemble, or a different framework) without modifying this module.

    Example:
        class MyAPITeacher(BaseTeacher): ...
        register_teacher("my_api", MyAPITeacher)
        teacher = load_teacher("my_api", api_key="...")
    """
    if not issubclass(cls, BaseTeacher):
        raise TypeError(f"{cls} must be a subclass of BaseTeacher")
    _REGISTRY[name] = cls
