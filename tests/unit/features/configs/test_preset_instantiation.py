"""BDD tests for KilatTransformer and KilatConfig preset instantiation."""
import pytest
import torch

from kilat.configs.model_config import KilatConfig
from kilat.models.transformer import KilatTransformer
from kilat.benchmarking.registry import build_model


class TestKilatPresets:
    def test_canonical_macro_synergy(self):
        """kilat-macro-synergy should default to recall_ratio=0.5 and macro_synergy pattern."""
        model = KilatTransformer.from_preset("kilat-macro-synergy")
        assert isinstance(model, KilatTransformer)
        assert model.config.recall_ratio == 0.5
        assert model.config.recall_pattern == "macro_synergy"
        assert model.config.use_rope is True

        # Forward pass verification
        x = torch.randint(0, model.config.vocab_size, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 8, model.config.vocab_size)

    def test_canonical_pure_attention(self):
        """kilat-pure-attention should instantiate with recall_ratio=1.0."""
        model = KilatTransformer.from_preset("kilat-pure-attention")
        assert model.config.recall_ratio == 1.0
        assert model.config.recall_pattern == "uniform"

        x = torch.randint(0, model.config.vocab_size, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 8, model.config.vocab_size)

    def test_canonical_pure_decay(self):
        """kilat-pure-decay should instantiate with recall_ratio=0.0."""
        model = KilatTransformer.from_preset("kilat-pure-decay")
        assert model.config.recall_ratio == 0.0
        assert model.config.recall_pattern == "uniform"

        x = torch.randint(0, model.config.vocab_size, (2, 8))
        out = model(x)
        assert out.logits.shape == (2, 8, model.config.vocab_size)

    def test_micro_and_mini_presets(self):
        """kilat-micro and kilat-mini presets should instantiate with correct dimensions."""
        micro = KilatTransformer.from_preset("kilat-micro")
        assert micro.config.n_embd == 64
        assert micro.config.n_layer == 2

        mini = KilatTransformer.from_preset("kilat-mini")
        assert mini.config.n_embd == 256
        assert mini.config.n_layer == 4

    def test_yaml_preset_loading(self):
        """Preset loader should resolve existing YAML presets such as small_dense."""
        model = KilatTransformer.from_preset("small_dense", n_layer=1, vocab_size=128)
        assert model.config.n_layer == 1
        assert model.config.vocab_size == 128
        assert model.config.n_embd == 768

    def test_hyperparameter_overrides_and_aliases(self):
        """Alias normalization (d_model -> n_embd, n_layers -> n_layer) must be respected."""
        cfg = KilatConfig.from_preset(
            "kilat-macro-synergy",
            d_model=128,
            n_heads=8,
            n_layers=3,
            vocab_size=256,
        )
        assert cfg.n_embd == 128
        assert cfg.n_head == 8
        assert cfg.n_layer == 3
        assert cfg.vocab_size == 256

    def test_unknown_preset_raises_key_error(self):
        """Non-existent preset should raise KeyError with helpful available presets."""
        with pytest.raises(KeyError, match="Unknown preset"):
            KilatTransformer.from_preset("non_existent_preset_xyz")

    def test_registry_integration_and_backward_pass(self):
        """build_model in benchmarking registry must route correctly and support backpropagation."""
        wrapped_model = build_model("kilat-macro-synergy", d_model=64, n_heads=4, n_layers=2)
        x = torch.randint(0, 64, (2, 8))
        logits = wrapped_model(x)
        assert logits.shape == (2, 8, 64)

        loss = logits.sum()
        loss.backward()

        # Invariant: Every trainable parameter must have a finite gradient
        for name, param in wrapped_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} missing gradient!"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient!"
