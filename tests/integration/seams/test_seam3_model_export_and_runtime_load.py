"""
Seam 3 Integration Test:
Verifies model checkpoint serialization, metadata export, and reload without weight or prediction drift.
Follows standards from ml_integration_testing_structure_md.md (Seam 3).
"""

import tempfile
from pathlib import Path
import numpy as np
import torch

from kilat.models.transformer import KilatTransformer
from kilat.configs.model_config import KilatConfig
from tests.shared.bdd import Scenario, Given, When, Then, And


def test_seam3_checkpoint_export_and_reload():
    with Scenario("Integration: Checkpoint Export and Runtime Reload Parity (Seam 3)"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "checkpoint_v1"

            with Given("an initialized KilatTransformer model with deterministic weights"):
                torch.manual_seed(42)
                config = KilatConfig(
                    vocab_size=500,
                    n_embd=64,
                    n_layer=2,
                    n_head=4,
                    recall_ratio=0.5,
                    ffn_mode="dense",
                )
                original_model = KilatTransformer(config)
                original_model.eval()

                golden_input = torch.randint(0, 500, (2, 8))
                with torch.no_grad():
                    baseline_logits = original_model(golden_input).logits.numpy()

                with When("the model checkpoint is saved to disk and reloaded into a fresh runtime session"):
                    original_model.save_pretrained(str(save_dir))
                    reloaded_model = KilatTransformer.from_pretrained(str(save_dir))
                    reloaded_model.eval()

                    with torch.no_grad():
                        reloaded_logits = reloaded_model(golden_input).logits.numpy()

                    with Then("the reloaded model output matches the baseline shape [2, 8, 500]"):
                        assert reloaded_logits.shape == baseline_logits.shape

                    with And("numerical drift between original and reloaded model is zero within tolerance (1e-5)"):
                        np.testing.assert_allclose(
                            reloaded_logits,
                            baseline_logits,
                            rtol=1e-5,
                            atol=1e-5,
                            err_msg="Reloaded checkpoint predictions deviate from baseline.",
                        )
