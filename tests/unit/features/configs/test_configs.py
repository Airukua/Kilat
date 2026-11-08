"""
Unit tests for Kilat Configuration Hierarchy.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import tempfile
import warnings
from pathlib import Path
import pytest

from kilat.configs.base import dump_yaml_file, load_yaml_file
from kilat.configs.model_config import KilatConfig
from kilat.configs.tokenizer_config import TokenizerConfig
from kilat.configs.dataloader_config import DataLoaderConfig
from kilat.configs.training_config import TrainingConfig
from kilat.configs.main_config import MainConfig
from kilat.utils.base_model import BaseConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_base_config_and_yaml_io():
    with Scenario("Unit: Base Configuration Serialization & IO"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_config.yaml"

            with Given("a sample dictionary configuration"):
                sample_data = {"name": "kilat_base", "version": 1, "nested": {"key": "val"}}

                with When("dump_yaml_file writes dictionary to a YAML file"):
                    dump_yaml_file(sample_data, file_path)
                    with Then("the file exists on disk"):
                        assert file_path.exists()

                with When("load_yaml_file reads the serialized configuration"):
                    loaded = load_yaml_file(file_path)
                    with Then("the restored data matches the original content"):
                        assert loaded == sample_data

            with Given("an instance of BaseConfig subclass"):
                class MockConfig(BaseConfig):
                    def __init__(self, a=10, b="kilat"):
                        super().__init__()
                        self.a = a
                        self.b = b

                cfg = MockConfig(a=20, b="test")
                with When("exporting configuration to dictionary"):
                    d = cfg.to_dict()
                    with Then("all attributes are preserved"):
                        assert d["a"] == 20
                        assert d["b"] == "test"


def test_kilat_model_config():
    with Scenario("Unit: Kilat Model Hyperparameters Specification"):
        with Given("default KilatConfig initialization"):
            cfg = KilatConfig()
            with When("inspecting default structural hyperparameters"):
                with Then("vocab_size defaults to 32000"):
                    assert cfg.vocab_size == 32000
                with And("n_embd defaults to 768"):
                    assert cfg.n_embd == 768
                with And("n_layer defaults to 12"):
                    assert cfg.n_layer == 12
                with And("n_head defaults to 12"):
                    assert cfg.n_head == 12
                with And("recall_ratio defaults to 0.5"):
                    assert cfg.recall_ratio == 0.5

        with Given("custom valid KilatConfig overrides"):
            with When("custom parameters are passed"):
                custom_cfg = KilatConfig(
                    vocab_size=16000,
                    n_embd=512,
                    n_layer=6,
                    n_head=8,
                    recall_ratio=0.25,
                )
                with Then("the overridden values are retained"):
                    assert custom_cfg.vocab_size == 16000
                    assert custom_cfg.n_embd == 512
                    assert custom_cfg.n_layer == 6
                    assert custom_cfg.n_head == 8

        with Given("invalid KilatConfig parameter combinations"):
            with When("n_embd is not divisible by n_head"):
                with Then("a ValueError is raised"):
                    with pytest.raises(ValueError):
                        KilatConfig(n_embd=500, n_head=7)

            with When("recall_ratio is negative or strictly greater than 1.0"):
                with And("a ValueError is raised"):
                    with pytest.raises(ValueError):
                        KilatConfig(recall_ratio=-0.1)
                    with pytest.raises(ValueError):
                        KilatConfig(recall_ratio=1.5)


def test_tokenizer_and_dataloader_configs():
    with Scenario("Unit: Tokenizer & DataLoader Configuration Contracts"):
        with Given("default TokenizerConfig"):
            t_cfg = TokenizerConfig()
            with When("checking default special token attributes"):
                with Then("tokenizer_type is gpt2"):
                    assert t_cfg.tokenizer_type == "gpt2"
                with And("use_fast flag is a boolean"):
                    assert isinstance(t_cfg.use_fast, bool)

        with Given("default DataLoaderConfig"):
            dl_cfg = DataLoaderConfig()
            with When("checking default data loader flags"):
                with Then("train_batch_size is positive"):
                    assert dl_cfg.train_batch_size > 0
                with And("num_workers is non-negative"):
                    assert dl_cfg.num_workers >= 0
                with And("pin_memory is a boolean flag"):
                    assert isinstance(dl_cfg.pin_memory, bool)

            with When("testing YAML string serialization of DataLoaderConfig"):
                yaml_str = dl_cfg.to_yaml()
                with Then("the YAML string is non-empty"):
                    assert len(yaml_str) > 0
                    assert "train_batch_size" in yaml_str

            with When("passing invalid arguments to DataLoaderConfig"):
                with Then("ValueErrors are properly raised"):
                    with pytest.raises(ValueError):
                        DataLoaderConfig(num_workers=-1)
                    with pytest.raises(ValueError):
                        DataLoaderConfig(train_batch_size=0)
                    with pytest.raises(ValueError):
                        DataLoaderConfig(max_seq_length=0)


def test_training_and_main_config_hierarchy():
    with Scenario("Unit: Training & Main Config Hierarchy Aggregation"):
        with Given("default TrainingConfig with fp32 precision for CPU"):
            train_cfg = TrainingConfig(precision="fp32")
            with When("checking optimization settings"):
                with Then("learning_rate is positive float"):
                    assert train_cfg.learning_rate > 0.0
                with And("weight_decay is non-negative float"):
                    assert train_cfg.weight_decay >= 0.0

            with When("testing YAML string serialization of TrainingConfig"):
                t_yaml = train_cfg.to_yaml()
                with Then("the serialized YAML is non-empty"):
                    assert len(t_yaml) > 0
                    assert "learning_rate" in t_yaml

            with When("passing invalid arguments to TrainingConfig"):
                with Then("ValueErrors are properly raised"):
                    with pytest.raises(ValueError):
                        TrainingConfig(training_mode="unknown_mode", precision="fp32")
                    with pytest.raises(ValueError):
                        TrainingConfig(training_mode="steps", max_steps=-1, precision="fp32")
                    with pytest.raises(ValueError):
                        TrainingConfig(training_mode="epochs", num_train_epochs=0, precision="fp32")
                    with pytest.raises(ValueError):
                        TrainingConfig(precision="invalid_prec")

        with Given("a fully assembled MainConfig with fp32 precision"):
            main_cfg = MainConfig(
                model=KilatConfig(),
                tokenizer=TokenizerConfig(),
                training=TrainingConfig(precision="fp32"),
                dataloader=DataLoaderConfig(),
            )
            with When("verifying aggregated child configuration sections"):
                with Then("model section contains KilatConfig"):
                    assert isinstance(main_cfg.model, KilatConfig)
                with And("tokenizer section contains TokenizerConfig"):
                    assert isinstance(main_cfg.tokenizer, TokenizerConfig)
                with And("dataloader section contains DataLoaderConfig"):
                    assert isinstance(main_cfg.dataloader, DataLoaderConfig)
                with And("training section contains TrainingConfig"):
                    assert isinstance(main_cfg.training, TrainingConfig)

            with When("saving MainConfig to YAML file and restoring it"):
                with tempfile.TemporaryDirectory() as tmp_yaml_dir:
                    yaml_target = Path(tmp_yaml_dir) / "experiment.yaml"
                    main_cfg.to_yaml(yaml_target)
                    with Then("the YAML configuration file is created"):
                        assert yaml_target.exists()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        restored = MainConfig.from_yaml(yaml_target)
                    with And("all sub-configs are faithfully restored"):
                        assert restored.vocab_size == main_cfg.vocab_size
                        assert restored.learning_rate == main_cfg.learning_rate
                        assert restored.n_layer == main_cfg.n_layer

            with When("testing convenience properties and build methods on MainConfig"):
                with Then("all proxy properties delegate to respective sub-configs"):
                    assert main_cfg.n_embd == 768
                    assert main_cfg.n_head == 12
                    assert main_cfg.pad_token_id == 0
                    assert main_cfg.eos_token_id == 2
                    assert main_cfg.bos_token_id == 1
                    assert main_cfg.precision == "fp32"
                    assert main_cfg.seed == 42
                    assert "MainConfig" in repr(main_cfg)

                built_model_cfg = main_cfg.build_model()
                with And("build_model returns valid KilatConfig"):
                    assert isinstance(built_model_cfg, KilatConfig)

                with tempfile.TemporaryDirectory() as save_dir:
                    main_cfg.save_pretrained(save_dir)
                    with And("save_pretrained creates config.yaml"):
                        assert (Path(save_dir) / "config.yaml").exists()


def test_main_config_extended_properties_and_factories():
    with Scenario("Unit: MainConfig Extended Delegation Properties and Factory Methods"):
        with Given("a MainConfig configured with specific values"):
            m_cfg = KilatConfig(
                vocab_size=1000,
                n_embd=64,
                n_layer=2,
                n_head=4,
                use_cache=True,
                recall_ratio=0.3,
                latent_dim=32,
                attn_drop=0.1,
                use_rope=True,
                rope_head_dim=16,
                rope_base=10000.0,
                ffn_mode="moe",
                ff_mult=2.0,
                ffn_dropout=0.05,
                num_experts=4,
                active_experts=2,
                num_shared_experts=1,
                fine_grained_factor=2,
                aux_loss_coef=0.01,
                device_balance_coef=0.02,
                embd_drop=0.05,
                resid_drop=0.05,
                initializer_range=0.02,
                tie_word_embeddings=True,
            )
            t_cfg = TokenizerConfig(tokenizer_type="gpt2")
            d_cfg = DataLoaderConfig(
                train_batch_size=8,
                eval_batch_size=16,
                num_workers=2,
                pin_memory=False,
                prefetch_factor=2,
                persistent_workers=False,
                drop_last=True,
                max_seq_length=128,
                truncation="right",
                use_packing=True,
                packed_block_size=64,
                use_distributed_sampler=False,
                distributed_shuffle=False,
                train_data_path="train.parquet",
                eval_data_path="eval.parquet",
                dataset_format="parquet",
                cache_dir="cache/",
                prefetch_batches=4,
            )
            tr_cfg = TrainingConfig(
                output_dir="outputs/",
                resume_from_checkpoint="ckpt/",
                save_checkpoints=True,
                atomic_checkpoint=True,
                training_mode="steps",
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-8,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                gradient_accumulation_steps=2,
                weight_decay=0.01,
                max_grad_norm=1.0,
                max_steps=500,
                num_train_epochs=3,
                warmup_steps=50,
                scheduler_type="cosine",
                scheduler_kwargs={"min_lr": 1e-6},
                logging_steps=10,
                eval_steps=50,
                save_steps=100,
                save_total_limit=3,
                early_stopping_patience=5,
                early_stopping_threshold=0.01,
                metric_for_best_model="loss",
                greater_is_better=False,
                precision="fp32",
                report_to=["none"],
                run_name="exp1",
                seed=123,
            )
            main = MainConfig(
                model=m_cfg,
                tokenizer=t_cfg,
                dataloader=d_cfg,
                training=tr_cfg,
                experiment_name="test-exp",
                description="desc",
                tags=["a", "b"],
            )

            with When("accessing all model delegation properties"):
                with Then("model attributes match expected values"):
                    assert main.use_cache is True
                    assert main.recall_ratio == 0.3
                    assert main.latent_dim == 32
                    assert main.attn_drop == 0.1
                    assert main.use_rope is True
                    assert main.rope_head_dim == 16
                    assert main.rope_base == 10000.0
                    assert main.ffn_mode == "moe"
                    assert main.ff_mult == 2.0
                    assert main.ffn_dropout == 0.05
                    assert main.num_experts == 4
                    assert main.active_experts == 2
                    assert main.num_shared_experts == 1
                    assert main.fine_grained_factor == 2
                    assert main.aux_loss_coef == 0.01
                    assert main.device_balance_coef == 0.02
                    assert main.embd_drop == 0.05
                    assert main.resid_drop == 0.05
                    assert main.initializer_range == 0.02
                    assert main.tie_word_embeddings is True

            with When("accessing all dataloader delegation properties"):
                with Then("dataloader attributes match expected values"):
                    assert main.train_batch_size == 8
                    assert main.eval_batch_size == 16
                    assert main.num_workers == 2
                    assert main.pin_memory is False
                    assert main.prefetch_factor == 2
                    assert main.persistent_workers is False
                    assert main.drop_last is True
                    assert main.max_seq_length == 128
                    assert main.truncation == "right"
                    assert main.use_packing is True
                    assert main.packed_block_size == 64
                    assert main.use_distributed_sampler is False
                    assert main.distributed_shuffle is False
                    assert main.train_data_path == "train.parquet"
                    assert main.eval_data_path == "eval.parquet"
                    assert main.dataset_format == "parquet"
                    assert main.cache_dir == "cache/"
                    assert main.prefetch_batches == 4

            with When("accessing all training delegation properties"):
                with Then("training attributes match expected values"):
                    assert main.output_dir == "outputs/"
                    assert main.resume_from_checkpoint == "ckpt/"
                    assert main.save_checkpoints is True
                    assert main.atomic_checkpoint is True
                    assert main.training_mode == "steps"
                    assert main.learning_rate == 1e-4
                    assert main.beta1 == 0.9
                    assert main.beta2 == 0.98
                    assert main.epsilon == 1e-8
                    assert main.per_device_train_batch_size == 8
                    assert main.per_device_eval_batch_size == 16
                    assert main.gradient_accumulation_steps == 2
                    assert main.weight_decay == 0.01
                    assert main.max_grad_norm == 1.0
                    assert main.max_steps == 500
                    assert main.num_train_epochs == 3
                    assert main.warmup_steps == 50
                    assert main.scheduler_type == "cosine"
                    assert main.scheduler_kwargs == {"min_lr": 1e-6}
                    assert main.logging_steps == 10
                    assert main.eval_steps == 50
                    assert main.save_steps == 100
                    assert main.save_total_limit == 3
                    assert main.early_stopping_patience == 5
                    assert main.early_stopping_threshold == 0.01
                    assert main.metric_for_best_model == "loss"
                    assert main.greater_is_better is False
                    assert main.precision == "fp32"
                    assert main.report_to == ["none"]
                    assert main.run_name == "exp1"
                    assert main.seed == 123

            with When("testing from_yaml_string"):
                with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
                    yaml_content = main.to_yaml(tmp.name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        restored = MainConfig.from_yaml_string(yaml_content)
                    with Then("restored config matches original"):
                        assert restored.experiment_name == "test-exp"
                        assert restored.vocab_size == 1000

            with When("testing from_main_config helper"):
                with Then("extracts KilatConfig from MainConfig, dict, or object"):
                    kc1 = MainConfig.from_main_config(main)
                    assert isinstance(kc1, KilatConfig)
                    assert kc1.vocab_size == 1000

                    kc2 = MainConfig.from_main_config({"model": m_cfg.to_dict()})
                    assert isinstance(kc2, KilatConfig)
                    assert kc2.vocab_size == 1000

                    kc3 = MainConfig.from_main_config(m_cfg.to_dict())
                    assert isinstance(kc3, KilatConfig)

                    with pytest.raises(TypeError):
                        MainConfig.from_main_config(12345)

            with When("testing from_yaml with missing sections"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_p = Path(tmp_dir)

                    dump_yaml_file({"model": {}}, tmp_p / "no_tok.yaml")
                    with Then("missing tokenizer raises ValueError"):
                        with pytest.raises(ValueError, match="tokenizer"):
                            MainConfig.from_yaml(tmp_p / "no_tok.yaml")

                    dump_yaml_file({"model": {}, "tokenizer": {}}, tmp_p / "no_dl.yaml")
                    with And("missing dataloader raises ValueError"):
                        with pytest.raises(ValueError, match="dataloader"):
                            MainConfig.from_yaml(tmp_p / "no_dl.yaml")

                    dump_yaml_file({"model": {}, "tokenizer": {}, "dataloader": {}}, tmp_p / "no_train.yaml")
                    with And("missing training raises ValueError"):
                        with pytest.raises(ValueError, match="training"):
                            MainConfig.from_yaml(tmp_p / "no_train.yaml")
