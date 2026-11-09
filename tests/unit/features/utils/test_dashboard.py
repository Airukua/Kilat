"""
Unit tests for Kilat Cyberpunk Terminal Dashboard (KilatDashboard).
"""

from __future__ import annotations

import logging
from rich.console import Console
from kilat.utils.dashboard import DashboardLogHandler, KilatDashboard


def test_dashboard_initialization() -> None:
    """Verify KilatDashboard initializes with expected attributes and safe fallbacks."""
    console = Console(width=120, force_terminal=True)
    dash = KilatDashboard(
        total_steps=5000,
        initial_step=100,
        learning_rate=3e-4,
        batch_size=32,
        optimizer_name="AdamW (β₁=0.9, β₂=0.98)",
        weight_decay=0.01,
        grad_clip=1.0,
        mixed_precision="BF16",
        tf32=True,
        cuda_device="NVIDIA GeForce RTX 4050",
        console=console,
    )

    assert dash.total_steps == 5000
    assert dash.current_step == 100
    assert dash.learning_rate == 3e-4
    assert dash.batch_size == 32
    assert dash.mixed_precision == "BF16"
    assert dash.tf32 is True
    assert len(dash.recent_logs) >= 1


def test_dashboard_update_and_metrics() -> None:
    """Verify update steps compute perplexity, trend arrow, and record curve history."""
    console = Console(width=120, force_terminal=True)
    dash = KilatDashboard(total_steps=1000, console=console)

    dash.update(step=50, loss=2.50, lr=5.0e-4, ppl=12.18, tokens=1024)
    assert dash.current_step == 50
    assert dash.train_loss == 2.50
    assert dash.perplexity == 12.18

    # Second step with decreasing loss (should have negative loss_delta)
    dash.update(step=100, loss=2.10, lr=4.8e-4, ppl=8.16, tokens=1024)
    assert dash.current_step == 100
    assert dash.loss_delta < 0  # decreasing loss
    assert len(dash.curve_history) >= 1


def test_dashboard_log_interception() -> None:
    """Verify DashboardLogHandler captures log messages into the recent_logs buffer."""
    console = Console(width=120, force_terminal=True)
    dash = KilatDashboard(total_steps=100, console=console)

    logger = logging.getLogger("test_kilat_logger")
    handler = DashboardLogHandler(dash)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Saved checkpoint-500 successfully.")
    assert any("Saved checkpoint-500" in log_entry for log_entry in dash.recent_logs)

    logger.removeHandler(handler)


def test_dashboard_render_output() -> None:
    """Verify render() produces all panels without raising exceptions."""
    console = Console(force_terminal=True)
    dash = KilatDashboard(total_steps=1000, console=console)
    dash.update(step=250, loss=1.85, lr=2.5e-4, ppl=6.35, tokens=2048)

    renderable = dash.render()
    assert renderable is not None

    # Test rendering into string
    with console.capture() as capture:
        console.print(renderable)
    output = capture.get()

    assert "KILAT" in output
    assert "Training Overview" in output
    assert "Optimization" in output
    assert "Training Curves" in output
    assert "System Monitor" in output
    assert "Status" in output
    assert "Recent Logs" in output
    assert "Building Efficient LLMs" in output


def test_dashboard_stop_cleanup() -> None:
    """Verify stop() cleans up Live context and NVML handle gracefully."""
    console = Console(width=120, force_terminal=True)
    dash = KilatDashboard(total_steps=100, console=console)
    dash.stop()
    assert dash.live is None
    assert dash._log_handler is None
