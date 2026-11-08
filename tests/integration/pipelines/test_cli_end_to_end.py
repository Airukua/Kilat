"""
End-to-End Integration Tests for Unified Kilat CLI (`kilat`) & Tiered Configs (Phase 6).

Exercises all 6 CLI subcommands (`prepare-data`, `verify`, `chaos`, `train` with `--set`,
`generate`, and `export-hf`) on real PyTorch micro-models and real on-disk artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kilat.cli import main as kilat_main
from kilat.configs.main_config import MainConfig
from tests.shared.bdd import And, Given, Scenario, Then, When


def test_tiered_configs_load_cleanly() -> None:
    with Scenario("Integration: Tiered YAML Configurations Parse Without Errors"):
        with Given("the repository configs/ directory"):
            repo_root = Path(__file__).resolve().parents[3]
            debug_yaml = repo_root / "configs" / "debug_cpu.yaml"
            dense_yaml = repo_root / "configs" / "small_dense.yaml"
            moe_yaml = repo_root / "configs" / "small_moe.yaml"
            std_moe_yaml = repo_root / "configs" / "standard_moe.yaml"

            with When("loading debug_cpu.yaml, small_dense.yaml, small_moe.yaml, and standard_moe.yaml"):
                cfg_debug = MainConfig.from_yaml(debug_yaml)
                # Suppress optional offline gpt2 resolution warning on CI machines without HF cache
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    cfg_dense = MainConfig.from_yaml(dense_yaml)
                    cfg_moe = MainConfig.from_yaml(moe_yaml)
                    cfg_std_moe = MainConfig.from_yaml(std_moe_yaml)

                with Then("each config preserves exact architectural and training settings"):
                    assert cfg_debug.model.vocab_size == 256
                    assert cfg_debug.dataloader.dataset_format == "synthetic"
                    assert cfg_debug.training.precision == "fp32"
                    assert cfg_dense.model.ffn_mode == "swiglu"
                    assert cfg_moe.model.ffn_mode == "moe"
                    assert cfg_moe.model.num_shared_experts == 1
                    assert cfg_std_moe.model.ffn_mode == "moe"
                    assert cfg_std_moe.model.num_experts == 64
                    assert cfg_std_moe.model.num_shared_experts == 2


def test_cli_end_to_end_full_lifecycle(tmp_path: Path, capsys: object) -> None:
    with Scenario("Integration: End-to-End Kilat CLI Lifecycle (6 Subcommands)"):
        repo_root = Path(__file__).resolve().parents[3]
        debug_yaml = str(repo_root / "configs" / "debug_cpu.yaml")

        with Given("a raw JSONL corpus on disk"):
            raw_jsonl = tmp_path / "raw_corpus.jsonl"
            records = [
                {"text": f"Kilat hybrid decay and latent MLA sample {i} " * 4}
                for i in range(12)
            ]
            raw_jsonl.write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8",
            )
            memmap_bin = tmp_path / "train.bin"

            with When("executing `kilat prepare-data --json`"):
                rc_prep = kilat_main(
                    [
                        "prepare-data",
                        "--input",
                        str(raw_jsonl),
                        "--output",
                        str(memmap_bin),
                        "--tokenizer",
                        "byte",
                        "--dtype",
                        "uint16",
                        "--json",
                    ]
                )
                out_prep = capsys.readouterr().out  # type: ignore[attr-defined]
                prep_meta = json.loads(out_prep)

                with Then("the memmap .bin and .meta.json are written cleanly with exit code 0"):
                    assert rc_prep == 0
                    assert memmap_bin.is_file()
                    assert (tmp_path / "train.bin.meta.json").is_file()
                    assert int(prep_meta["num_tokens"]) > 100

            with And("executing `kilat verify --config configs/debug_cpu.yaml` against the memmap"):
                verify_dir = tmp_path / "verify_out"
                rc_verify = kilat_main(
                    [
                        "verify",
                        "--config",
                        debug_yaml,
                        "--output-dir",
                        str(verify_dir),
                        "--memmap-path",
                        str(memmap_bin),
                        "--memmap-dtype",
                        "uint16",
                        "--json",
                    ]
                )
                out_verify = capsys.readouterr().out  # type: ignore[attr-defined]
                verify_json = json.loads(out_verify)

                assert rc_verify == 0
                assert verify_json["overall_status"] == "PASS"
                assert (verify_dir / "assurance_report.json").is_file()
                assert (verify_dir / "assurance_report.md").is_file()

            with And("executing `kilat chaos --scenarios C-01,C-08 --json`"):
                rc_chaos = kilat_main(
                    [
                        "chaos",
                        "--scenarios",
                        "C-01,C-08",
                        "--json",
                    ]
                )
                out_chaos = capsys.readouterr().out  # type: ignore[attr-defined]
                chaos_json = json.loads(out_chaos)

                assert rc_chaos == 0
                assert chaos_json["overall_status"] == "PASS"
                assert chaos_json["scenarios_caught"] == 2

            with And("executing `kilat train` with dot-notation `--set` overrides"):
                ckpt_root = tmp_path / "train_runs"
                rc_train = kilat_main(
                    [
                        "train",
                        "--config",
                        debug_yaml,
                        "--set",
                        f"training.output_dir={ckpt_root}",
                        "--set",
                        "training.max_steps=4",
                        "--set",
                        "training.logging_steps=2",
                        "--set",
                        "training.save_steps=4",
                        "--set",
                        "training.eval_steps=4",
                        "--quiet",
                    ]
                )
                capsys.readouterr()  # type: ignore[attr-defined]

                assert rc_train == 0
                ckpt_final = ckpt_root / "checkpoint-final"
                assert ckpt_final.is_dir()
                assert (ckpt_final / "checkpoint_manifest.json").is_file()
                assert (ckpt_root / "telemetry.jsonl").is_file()

            with And("executing `kilat generate` from the trained checkpoint"):
                rc_gen = kilat_main(
                    [
                        "generate",
                        "--checkpoint",
                        str(ckpt_final),
                        "--prompt",
                        "Kilat",
                        "--max-new-tokens",
                        "6",
                        "--json",
                    ]
                )
                out_gen = capsys.readouterr().out  # type: ignore[attr-defined]
                gen_json = json.loads(out_gen)

                assert rc_gen == 0
                assert isinstance(gen_json["generated_text"], str)
                assert len(gen_json["generated_text"]) > 0

            with And("executing `kilat export-hf` to produce a standalone HF directory"):
                hf_dir = tmp_path / "hf_export"
                rc_exp = kilat_main(
                    [
                        "export-hf",
                        "--checkpoint",
                        str(ckpt_final),
                        "--output",
                        str(hf_dir),
                        "--json",
                    ]
                )
                out_exp = capsys.readouterr().out  # type: ignore[attr-defined]
                exp_json = json.loads(out_exp)

                assert rc_exp == 0
                assert exp_json["success"] is True
                assert (hf_dir / "config.json").is_file()
                assert (hf_dir / "model.safetensors").is_file() or (
                    hf_dir / "pytorch_model.bin"
                ).is_file()

            with And("verifying installed console entrypoint `kilat --version`"):
                proc = subprocess.run(
                    [sys.executable, "-m", "kilat.cli", "--version"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                assert proc.returncode == 0
                assert "kilat" in proc.stdout
