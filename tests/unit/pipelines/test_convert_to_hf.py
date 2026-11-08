"""
Unit tests for Kilat checkpoint to Hugging Face converter.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from kilat.inference.export_hf import (
    WeightFileNotFoundError,
    _clean_state_dict,
    convert_checkpoint_to_hf,
    copy_tokenizer_from_checkpoint,
    load_config_from_checkpoint,
    load_model_weights,
    save_tokenizer_config,
    validate_model,
)
from kilat.models.transformer import KilatTransformer
from kilat.configs.model_config import KilatConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_clean_state_dict_ddp_and_tying():
    with Scenario("Unit: Checkpoint State Dict Cleaning"):
        with Given("a state dict with DDP 'module.' prefix and weight tying"):
            weight_tensor = torch.randn(10, 16)
            raw_state_dict = {
                "module.wte.weight": weight_tensor,
                "module.layer.0.weight": torch.randn(16, 16),
            }

            with When("_clean_state_dict is executed"):
                cleaned = _clean_state_dict(raw_state_dict)

                with Then("the 'module.' prefix is stripped from keys"):
                    assert "wte.weight" in cleaned
                    assert "layer.0.weight" in cleaned
                    assert "module.wte.weight" not in cleaned

                with And("lm_head.weight is automatically populated via weight tying"):
                    assert "lm_head.weight" in cleaned
                    assert torch.equal(cleaned["lm_head.weight"], weight_tensor)


def test_convert_checkpoint_to_hf_execution():
    with Scenario("Unit: Full HF Checkpoint Conversion"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpt_dir = tmp_path / "kilat_ckpt"
            hf_out_dir = tmp_path / "hf_converted"

            with Given("a saved Kilat model checkpoint"):
                config = KilatConfig(
                    vocab_size=128,
                    n_embd=32,
                    n_layer=1,
                    n_head=2,
                    recall_ratio=0.5,
                    ffn_mode="dense",
                )
                model = KilatTransformer(config)
                model.save_pretrained(str(ckpt_dir))

                with When("convert_checkpoint_to_hf is executed"):
                    results = convert_checkpoint_to_hf(
                        checkpoint_dir=str(ckpt_dir),
                        output_dir=str(hf_out_dir),
                        use_safetensors=True,
                        device="cpu",
                        skip_tokenizer=True,
                    )

                    with Then("the conversion completes successfully"):
                        assert results["success"] is True

                    with And("model.safetensors and config.json are created in HF output directory"):
                        assert (hf_out_dir / "model.safetensors").exists()
                        assert (hf_out_dir / "config.json").exists()


def test_converter_helpers_and_validation():
    with Scenario("Unit: Converter Tokenizer Copying and Validation Functions"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpt_dir = tmp_path / "ckpt"
            out_dir = tmp_path / "out"
            ckpt_dir.mkdir()
            out_dir.mkdir()

            config = KilatConfig(vocab_size=64, n_embd=16, n_layer=1, n_head=2)
            model = KilatTransformer(config)

            with Given("tokenizer files in checkpoint"):
                (ckpt_dir / "tokenizer.json").write_text('{"version": "1.0"}')
                (ckpt_dir / "spm.model").write_bytes(b"dummy_spm")

                with When("copying tokenizer from checkpoint"):
                    copied = copy_tokenizer_from_checkpoint(ckpt_dir, out_dir, config)
                    with Then("tokenizer files are copied"):
                        assert copied is True
                        assert (out_dir / "tokenizer.json").exists()
                        assert (out_dir / "spm.model").exists()

                with When("saving tokenizer configuration fallback"):
                    save_tokenizer_config(out_dir, config, ckpt_dir)
                    with Then("tokenizer_config.json is generated in out directory"):
                        assert (out_dir / "tokenizer_config.json").exists()

            with When("validating model using validate_model"):
                val_res = validate_model(model, config, device="cpu", num_tokens=5)
                with Then("validation returns success"):
                    assert val_res["success"] is True

            with When("attempting load_model_weights on empty directory"):
                empty_dir = tmp_path / "empty"
                empty_dir.mkdir()
                with Then("WeightFileNotFoundError is raised"):
                    with pytest.raises(WeightFileNotFoundError):
                        load_model_weights(model, empty_dir)


def test_convert_checkpoint_with_custom_config_and_tokenizer():
    with Scenario("Unit: HF Converter with Custom Config Path and Tokenizer Integration"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpt_dir = tmp_path / "ckpt"
            out_dir = tmp_path / "hf_out"
            cfg_file = tmp_path / "custom_config.yaml"

            with Given("a saved model and separate custom config file"):
                config = KilatConfig(vocab_size=64, n_embd=16, n_layer=1, n_head=2)
                model = KilatTransformer(config)
                model.save_pretrained(str(ckpt_dir))
                config.to_yaml(cfg_file)

                # Add dummy tokenizer file to checkpoint
                (ckpt_dir / "tokenizer.json").write_text('{"vocab": {}}')

                with When("converting using explicit config_path and skip_tokenizer=False"):
                    res = convert_checkpoint_to_hf(
                        checkpoint_dir=str(ckpt_dir),
                        output_dir=str(out_dir),
                        config_path=str(cfg_file),
                        skip_tokenizer=False,
                        skip_validation=False,
                    )
                    with Then("the conversion succeeds"):
                        assert res["success"] is True
                        assert (out_dir / "tokenizer.json").exists()
                        assert res["tokenizer_copied"] is True


def test_converter_config_and_weight_priorities():
    import json
    from kilat.inference.export_hf import ConversionError, ValidationError

    with Scenario("Unit: HF Converter Config and Weight Priority Escalation"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            with Given("a checkpoint with corrupted config.yaml and valid config.json"):
                corrupt_dir = tmp_path / "corrupt_yaml"
                corrupt_dir.mkdir()
                (corrupt_dir / "config.yaml").write_text(": invalid yaml [")
                (corrupt_dir / "config.json").write_text(json.dumps({"vocab_size": 256, "n_embd": 32, "n_layer": 1, "n_head": 2}))

                with When("loading config from corrupted yaml directory"):
                    cfg = load_config_from_checkpoint(corrupt_dir)
                    with Then("it falls back to config.json"):
                        assert cfg.vocab_size == 256

            with Given("a checkpoint with only training_args.json"):
                args_dir = tmp_path / "args_dir"
                args_dir.mkdir()
                (args_dir / "training_args.json").write_text(json.dumps({"model_config": {"vocab_size": 512, "n_embd": 64, "n_layer": 2, "n_head": 4}}))

                with When("loading config from training_args directory"):
                    cfg_args = load_config_from_checkpoint(args_dir)
                    with Then("it extracts model config from training_args.json"):
                        assert cfg_args.vocab_size == 512

            with Given("an empty checkpoint directory"):
                empty_dir = tmp_path / "empty_cfg"
                empty_dir.mkdir()
                with When("loading config from empty directory"):
                    default_cfg = load_config_from_checkpoint(empty_dir)
                    with Then("it falls back to default KilatConfig"):
                        assert default_cfg.vocab_size == 32000

            with Given("a model and checkpoint with training_state.pt weights"):
                ts_dir = tmp_path / "ts_dir"
                ts_dir.mkdir()
                cfg_small = KilatConfig(vocab_size=64, n_embd=16, n_layer=1, n_head=2)
                model = KilatTransformer(cfg_small)
                torch.save({"model_state_dict": model.state_dict()}, ts_dir / "training_state.pt")

                with When("loading weights from training_state.pt"):
                    loaded_m, meta = load_model_weights(model, ts_dir, device="cpu", strict=False)
                    with Then("weight source is identified as training_state.pt"):
                        assert meta["source"] == "training_state.pt"

                with When("loading weights with strict=True and invalid keys"):
                    invalid_ts_dir = tmp_path / "invalid_ts"
                    invalid_ts_dir.mkdir()
                    torch.save({"model_state_dict": {"unexpected.key": torch.zeros(1)}}, invalid_ts_dir / "training_state.pt")
                    with Then("WeightLoadingError is raised"):
                        from kilat.inference.export_hf import WeightLoadingError
                        with pytest.raises(WeightLoadingError):
                            load_model_weights(model, invalid_ts_dir, device="cpu", strict=True)

            with Given("convert_checkpoint_to_hf with json config_path"):
                json_cfg_path = tmp_path / "custom.json"
                json_cfg_path.write_text(json.dumps(cfg_small.to_dict()))
                hf_out2 = tmp_path / "hf_out2"

                with When("converting checkpoint with JSON config and verbose=True"):
                    res = convert_checkpoint_to_hf(
                        checkpoint_dir=str(ts_dir),
                        output_dir=str(hf_out2),
                        config_path=str(json_cfg_path),
                        use_safetensors=True,
                        skip_tokenizer=True,
                        verbose=True,
                    )
                    with Then("conversion succeeds"):
                        assert res["success"] is True

            with Given("an uninitializable checkpoint"):
                bad_dir = tmp_path / "bad_dir"
                bad_dir.mkdir()
                with When("convert_checkpoint_to_hf encounters missing weight files"):
                    with Then("ConversionError is raised"):
                        with pytest.raises(ConversionError):
                            convert_checkpoint_to_hf(bad_dir, tmp_path / "bad_out")
