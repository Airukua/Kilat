"""
Pytest configuration and global hooks for KILAT test suite.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root and src are discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.shared.helpers.ascii_logo import get_kilat_banner


def pytest_sessionstart(session):
    """Display the KILAT Claude-style ASCII logo at the start of the test run."""
    # Write directly to terminal stdout
    sys.stdout.write(get_kilat_banner(version="1.0.0") + "\n\n")
    sys.stdout.flush()


def pytest_report_teststatus(report, config):
    """Suppress default pytest dots to keep clean BDD terminal hierarchy output."""
    if report.when in ("call", "setup", "teardown"):
        return report.outcome, "", ""


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure deterministic random seed across tests as specified in ML testing standard."""
    import random
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
