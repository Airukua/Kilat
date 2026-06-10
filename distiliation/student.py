from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseStudent(ABC, nn.Module):
    """
    Interface for all student models in distillation.

    WHY: Student model is trainable (unlike teacher). It can be initialized from
    scratch or from a previous checkpoint. The abstraction allows swapping
    between different architectures (Kilat, HuggingFace, custom) without
    changing the distillation loop.

    Key design:
    - `load()`: resume from a checkpoint (e.g., after interruption).
    - `from_config()`: create a new untrained student.
    - `forward()` must return logits matching teacher's output shape.
    - No freezing by default – student is trainable.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> "BaseStudent":
        """Load student from an existing checkpoint (resume training)."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, *args, **kwargs) -> "BaseStudent":
        """Create a new student from a configuration (train from scratch)."""
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

    @property
    def vocab_size(self) -> int:
        """Vocabulary size; must match teacher's vocab for KL divergence."""
        raise NotImplementedError

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        try:
            n = self.num_parameters / 1e6
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
            return (
                f"{self.__class__.__name__}"
                f"({n:.1f}M params, {trainable:.1f}M trainable)"
            )
        except Exception:
            return f"{self.__class__.__name__}(trainable)"


class KilatStudent(BaseStudent):
    """
    Student based on KilatTransformer – typically smaller than the teacher.

    WHY: Using the same codebase as the teacher ensures compatible tokenizers,
    config handling, and checkpoint formats. This is the most common path for
    progressive distillation (Kilat teacher → Kilat student).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        ffn_mode: str = "dense",
        recall_ratio: float = 0.5,
        device: str | torch.device = "cpu",
        **extra_kwargs,
    ) -> "KilatStudent":
        """
        Create a new student from scratch with explicit architectural parameters.

        WHY: This is the primary way to define a small student before distillation.
        The arguments mirror KilatConfig fields, making it easy to shrink the model.

        Example:
            student = KilatStudent.from_config(
                vocab_size=50000,
                n_embd=256,
                n_layer=4,
                n_head=4,
            )
        """
        from arc.model import KilatTransformer
        from config import KilatConfig

        cfg = KilatConfig(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            ffn_mode=ffn_mode,
            recall_ratio=recall_ratio,
            **extra_kwargs,
        )
        model = KilatTransformer(cfg).to(device)
        return cls(model)

    @classmethod
    def from_config_obj(
        cls,
        config,
        device: str | torch.device = "cpu",
    ) -> "KilatStudent":
        """Create student from an existing KilatConfig or MainConfig object."""
        from arc.model import KilatTransformer
        from config import MainConfig

        if isinstance(config, MainConfig):
            config = config.model

        model = KilatTransformer(config).to(device)
        return cls(model)

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
    ) -> "KilatStudent":
        """
        Load a student from a previous distillation checkpoint.

        WHY: Supports both safetensors (preferred) and legacy .pt files.
        The directory must contain config.yaml and model weights.
        """
        from arc.model import KilatTransformer
        from config import MainConfig
        from safetensors.torch import load_file

        checkpoint_dir = Path(checkpoint_dir)
        main_cfg = MainConfig.from_yaml(str(checkpoint_dir / "config.yaml"))
        model = KilatTransformer(main_cfg.model)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            model.load_state_dict(load_file(str(safetensors_path), device=str(device)))
        else:
            pt_path = checkpoint_dir / "model.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
            model.load_state_dict(torch.load(str(pt_path), map_location=device))

        return cls(model.to(device))

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        # Normalize output: KilatTransformer may return (logits, loss) or a dict.
        return out if isinstance(out, torch.Tensor) else out.logits

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def config(self):
        return self.model.config


class HuggingFaceStudent(BaseStudent):
    """
    Student from a HuggingFace causal LM (e.g., GPT‑2, small LLaMA, DistilGPT2).

    WHY: Allows distilling into well‑known architectures that can later be served
    with standard HF pipelines. Also enables using pretrained checkpoints as a
    warm start before distillation fine‑tuning.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    @classmethod
    def from_config(
        cls,
        model_name_or_path: str,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
    ) -> "HuggingFaceStudent":
        """
        Load a pretrained HF model as student (then fine‑tune via distillation).

        WHY: This is the typical "distill into existing small model" scenario.
        The student already has reasonable weights; distillation further improves it.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)

        return cls(model)

    @classmethod
    def from_scratch(
        cls,
        config,
        device: str | torch.device = "cpu",
    ) -> "HuggingFaceStudent":
        """
        Create a student from a HF config without pretrained weights.

        WHY: Useful when you want complete control over initialization,
        e.g., for ablation studies or when pretrained weights are not available.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(config).to(device)
        return cls(model)

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "HuggingFaceStudent":
        """Load a HF student from a saved checkpoint directory."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch_dtype,
        ).to(device)
        return cls(model)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        # HF causal LM outputs have a .logits attribute.
        return out.logits

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size


_REGISTRY: dict[str, type[BaseStudent]] = {
    "kilat": KilatStudent,
    "huggingface": HuggingFaceStudent,
}

def load_student(backend: str, *args, **kwargs) -> BaseStudent:
    """
    Load a student from a checkpoint using a registered backend.

    WHY: Single factory function decouples training scripts from concrete classes.
    The user only needs to know the backend name (e.g., "kilat") and checkpoint path.

    Example:
        student = load_student("kilat", "checkpoints/student-v1", device="cuda")
        student = load_student("huggingface", "distilgpt2", device="cuda")
    """
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[backend].load(*args, **kwargs)

def build_student(backend: str, *args, **kwargs) -> BaseStudent:
    """
    Create a new student from a configuration (train from scratch).

    WHY: Symmetry with load_student. Build is used for initial creation;
    load is used for resuming.

    Example:
        student = build_student("kilat", vocab_size=50000, n_embd=256, n_layer=4, n_head=4)
        student = build_student("huggingface", "Qwen/Qwen2-0.5B")
    """
    if backend not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[backend].from_config(*args, **kwargs)

def register_student(name: str, cls: type[BaseStudent]):
    """
    Register a custom student class for use with the factory functions.

    WHY: Allows users to plug in custom architectures without forking this module.

    Example:
        class MyCustomStudent(BaseStudent): ...
        register_student("my_custom", MyCustomStudent)
        student = build_student("my_custom", ...)
    """
    if not issubclass(cls, BaseStudent):
        raise TypeError(f"{cls} must be a subclass of BaseStudent")
    _REGISTRY[name] = cls
