"""Module entrypoint for python -m kilat.benchmarking."""
import sys
from pathlib import Path

# Add repo root to path so tools/certify.py can be invoked
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root_dir / "tools"))

from certify import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
