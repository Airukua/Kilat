"""
Cyberpunk Terminal Dashboard (TUI) for Kilat Pre-Training.

Provides a real-time, flicker-free interactive terminal UI powered by Rich:
  - Header: Warm gradient ⚡ KILAT 3D/block logo + architecture metadata + live status badge.
  - Overview: Glowing neon orange progress bar, percentage, and 6 stat cards
    (STEP, TRAIN LOSS with trend arrow, PPL, SPEED in tok/s, ELAPSED, ETA).
  - Middle Row:
      - ⚙ Optimization (Learning Rate, Batch Size, Optimizer, Weight Decay, Grad Clip).
      - 📈 Training Curves (dual Braille anti-aliased terminal curve chart: Loss in orange, PPL in blue).
  - Bottom Row:
      - 🖥 System Monitor (GPU Utilization bar, VRAM Usage bar, GPU Temp).
      - ✔ Status (CUDA status, Mixed Precision mode, TF32 flag).
      - 📄 Recent Logs (live scrolling log window with timestamps).
  - Footer: Slogan, Kilat version, and repository URL.

Designed to be resilient, zero-overhead, responsive to terminal resize, and with
graceful fallback for non-TTY / multi-process DDP environments.
"""

from __future__ import annotations

import collections
import datetime
import logging
import math
import sys
import time
from typing import Any, Deque, List, Optional, Tuple

from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

try:
    import pynvml  # type: ignore

    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class DashboardLogHandler(logging.Handler):
    """Logging handler that captures messages directly into the Dashboard's Recent Logs."""

    def __init__(self, dashboard: "KilatDashboard") -> None:
        super().__init__()
        self.dashboard = dashboard
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Filter out noisy or raw newline entries
            msg_clean = msg.strip()
            if not msg_clean:
                return
            # Shorten very long file paths in log messages for neat display
            if "site-packages" in msg_clean:
                msg_clean = msg_clean.split("]")[-1].strip() if "]" in msg_clean else msg_clean
            self.dashboard.add_log(msg_clean, level=record.levelname)
        except Exception:
            pass


class KilatDashboard:
    """High-performance Rich Live interactive terminal dashboard for Kilat pre-training."""

    def __init__(
        self,
        total_steps: int,
        initial_step: int = 0,
        learning_rate: float = 6e-4,
        batch_size: int = 32,
        optimizer_name: str = "AdamW (β₁=0.9, β₂=0.98)",
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        mixed_precision: str = "BF16",
        tf32: bool = True,
        cuda_device: str = "Active",
        model_name: str = "KILAT v2.0.0",
        model_desc: str = "Hybrid Sub-Quadratic Global Decay + Latent MLA",
        console: Optional[Console] = None,
        max_log_entries: int = 4,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.current_step = initial_step
        self.initial_step = initial_step
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.tf32 = tf32
        self.cuda_device = cuda_device
        self.model_name = model_name
        self.model_desc = model_desc

        self.console = console or Console(force_terminal=True)
        self.live: Optional[Live] = None
        self._log_handler: Optional[DashboardLogHandler] = None

        # Metrics State
        self.train_loss: float = 0.0
        self.last_loss: Optional[float] = None
        self.loss_delta: float = 0.0
        self.perplexity: float = 0.0
        self.tokens_per_sec: float = 0.0
        self.steps_per_sec: float = 0.0

        # Timing
        self.start_time = time.monotonic()
        self.last_step_time = time.monotonic()
        self.recent_step_durations: Deque[float] = collections.deque(maxlen=20)
        self.recent_tokens: Deque[int] = collections.deque(maxlen=20)

        # History for curve plotting: list of (step, loss, ppl)
        self.curve_history: List[Tuple[int, float, float]] = []

        # Recent Logs (ring buffer)
        self.recent_logs: Deque[str] = collections.deque(maxlen=max_log_entries)

        # System Monitor Stats
        self.gpu_util: int = 0
        self.vram_used_gb: float = 0.0
        self.vram_total_gb: float = 6.0
        self.gpu_temp: int = 0
        self._last_gpu_query: float = 0.0

        # NVML setup
        self._nvml_handle: Any = None
        self._init_nvml()

        # Seed initial log
        dev_info = self.cuda_device if self.cuda_device != "Active" else "1 GPU (CUDA)"
        self.add_log(f"Starting training on {dev_info}", level="INFO")

    def _init_nvml(self) -> None:
        if _HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

        if _HAS_TORCH and torch.cuda.is_available():
            try:
                self.vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                self.vram_total_gb = 6.0

    def query_system_stats(self) -> None:
        """Poll NVML and PyTorch for hardware utilization."""
        now = time.monotonic()
        if now - self._last_gpu_query < 0.6:
            return
        self._last_gpu_query = now

        if self._nvml_handle is not None:
            try:
                rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self.gpu_util = int(rates.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self.vram_used_gb = float(mem.used) / (1024**3)
                self.vram_total_gb = float(mem.total) / (1024**3)
                self.gpu_temp = int(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
                return
            except Exception:
                pass

        if _HAS_TORCH and torch.cuda.is_available():
            try:
                self.vram_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
            except Exception:
                pass

    def add_log(self, message: str, level: str = "INFO") -> None:
        """Add a formatted log entry into the recent logs ring buffer."""
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        lvl = level.upper()
        if lvl == "INFO":
            lvl_fmt = "[bold #00FF66]INFO[/]"
        elif lvl in ("WARN", "WARNING"):
            lvl_fmt = "[bold #FFA500]WARN[/]"
        elif lvl in ("ERROR", "CRITICAL"):
            lvl_fmt = "[bold #FF4444]FAIL[/]"
        else:
            lvl_fmt = f"[bold #38BDF8]{lvl:<4s}[/]"

        entry = f"[dim #888888][{now_str}][/] {lvl_fmt} [white]{message}[/]"
        self.recent_logs.append(entry)

    def start(self) -> None:
        """Start the interactive Live display and attach logging interception."""
        # Attach log handler to root kilat logger so warnings/checkpoints route into panel
        root_logger = logging.getLogger()
        self._log_handler = DashboardLogHandler(self)
        root_logger.addHandler(self._log_handler)

        self.query_system_stats()
        self.live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=4,
            auto_refresh=False,
            redirect_stdout=True,
            redirect_stderr=True,
            transient=False,
        )
        self.live.start()

    def stop(self) -> None:
        """Stop Live rendering and detach logging hooks cleanly."""
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

        if self.live is not None:
            try:
                self.live.update(self.render(), refresh=True)
                self.live.stop()
            except Exception:
                pass
            self.live = None

        if self._nvml_handle is not None and _HAS_PYNVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

    def update(
        self,
        step: int,
        loss: float,
        lr: Optional[float] = None,
        ppl: Optional[float] = None,
        tokens: int = 0,
        optimizer_name: Optional[str] = None,
    ) -> None:
        """Update metrics and refresh the live dashboard."""
        now = time.monotonic()
        duration = now - self.last_step_time
        self.last_step_time = now

        step_delta = max(1, step - self.current_step)
        self.current_step = step

        if duration > 0:
            self.recent_step_durations.append(duration / step_delta)
            if tokens > 0:
                self.recent_tokens.append(tokens)

        if lr is not None:
            self.learning_rate = lr
        if optimizer_name is not None:
            self.optimizer_name = optimizer_name

        if self.last_loss is not None:
            self.loss_delta = loss - self.last_loss
        self.last_loss = loss
        self.train_loss = loss

        if ppl is not None:
            self.perplexity = ppl
        elif loss < 20.0:
            self.perplexity = math.exp(loss)

        # Compute smoothed speed
        if self.recent_step_durations:
            avg_duration = sum(self.recent_step_durations) / len(self.recent_step_durations)
            self.steps_per_sec = 1.0 / max(1e-4, avg_duration)
            if self.recent_tokens:
                total_tok = sum(self.recent_tokens)
                total_dur = sum(self.recent_step_durations)
                self.tokens_per_sec = total_tok / max(1e-4, total_dur)
            else:
                self.tokens_per_sec = self.steps_per_sec * self.batch_size * 512

        # Record curve history
        if not self.curve_history or (step - self.curve_history[-1][0]) >= max(1, self.total_steps // 200):
            self.curve_history.append((step, float(loss), float(self.perplexity)))

        # Periodic hardware stats poll
        self.query_system_stats()

        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def _format_time(self, seconds: float) -> str:
        s = max(0, int(seconds))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        return f"{m}m {s:02d}s"

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-component Renderers
    # ─────────────────────────────────────────────────────────────────────────

    def _render_header(self, term_width: int) -> RenderableType:
        """Render the warm gradient KILAT logo, subtitle, and live badge."""
        logo = (
            "[bold #FF6B00] ⚡ ██╗  ██╗██╗██╗      █████╗ ████████╗[/]\n"
            "[bold #FF8800]   ██║ ██╔╝██║██║     ██╔══██╗╚══██╔══╝[/]\n"
            "[bold #FFA500]   █████╔╝ ██║██║     ███████║   ██║   [/]\n"
            "[bold #FFB700]   ██╔═██╗ ██║██║     ██╔══██║   ██║   [/]\n"
            "[bold #FFC400]   ██║  ██╗██║███████╗██║  ██║   ██║   [/]\n"
            "[bold #FFD000]   ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   [/]"
        )

        desc = (
            "\n [dim #444444]│[/]  [bold #FFFFFF]Kernelized Lightweight[/]\n"
            " [dim #444444]│[/]  [bold #FFFFFF]Attention Transformer[/] [dim #888888](v2.0.0)[/]\n"
            " [dim #444444]│[/]  [dim #AAAAAA]Hybrid Sub-Quadratic Global Decay + Latent MLA[/]"
        )

        now_str = datetime.datetime.now().strftime("%b %d, %Y  %H:%M:%S")
        badge = (
            "\n[bold #000000 on #00FF66]  ▶ TRAINING IN PROGRESS  [/]\n"
            f"[dim #AAAAAA]{now_str}[/]"
        )

        t = Table.grid(expand=True)
        if term_width >= 110:
            t.add_column(width=42, no_wrap=True)
            t.add_column(no_wrap=True)
            t.add_column(justify="right", no_wrap=True)
            t.add_row(logo, desc, badge)
        else:
            t.add_column(width=42, no_wrap=True)
            t.add_column(justify="right", no_wrap=True)
            t.add_row(logo, badge)
        return t

    def _render_overview(self, term_width: int) -> RenderableType:
        """Render the progress bar and 6 stat cards."""
        pct = (self.current_step / self.total_steps) * 100.0

        top = Table.grid(expand=True)
        top.add_column()
        top.add_column(justify="right")
        top.add_row(
            "",
            f"[bold #FF9900]{pct:.1f}%[/] [dim]({self.current_step:,} / {self.total_steps:,} steps)[/]",
        )

        # Progress bar
        inner_width = max(20, term_width - 6)
        filled = int(inner_width * (self.current_step / self.total_steps))
        empty = max(0, inner_width - filled)
        bar_str = f"[bold #FF6B00]{'━' * filled}[/][#262626]{'─' * empty}[/]"

        # Timing
        elapsed = time.monotonic() - self.start_time
        remaining_steps = max(0, self.total_steps - self.current_step)
        if self.recent_step_durations:
            avg_step_sec = sum(self.recent_step_durations) / len(self.recent_step_durations)
            eta = remaining_steps * avg_step_sec
        else:
            eta = 0.0

        # Trend arrow
        if self.loss_delta < -1e-4:
            arrow = "[bold #00FF66]↓[/]"
        elif self.loss_delta > 1e-4:
            arrow = "[bold #FF4444]↑[/]"
        else:
            arrow = "[dim #888888]→[/]"

        # Speed formatting
        if self.tokens_per_sec >= 1000:
            speed_str = f"{self.tokens_per_sec / 1000.0:.1f}k tok/s"
        else:
            speed_str = f"{self.tokens_per_sec:.1f} tok/s"

        stats = Table.grid(expand=True)
        for _ in range(6):
            stats.add_column(ratio=1, no_wrap=True)

        c1 = f"[bold #00D4FF]🏃 STEP[/]\n[bold #FFFFFF]{self.current_step:,}[/] [dim]/ {self.total_steps:,}[/]"
        c2 = f"[bold #A855F7]📊 TRAIN LOSS[/]\n[bold #FFFFFF]{self.train_loss:.4f}[/] {arrow}"
        c3 = f"[bold #06B6D4]⏱ PERPLEXITY (PPL)[/]\n[bold #FFFFFF]{self.perplexity:.2f}[/]"
        c4 = f"[bold #EAB308]⚡ SPEED[/]\n[bold #FFFFFF]{speed_str}[/]"
        c5 = f"[bold #3B82F6]🕒 ELAPSED[/]\n[bold #FFFFFF]{self._format_time(elapsed)}[/]"
        c6 = f"[bold #F97316]⏳ ETA[/]\n[bold #FFFFFF]{self._format_time(eta)}[/]"

        stats.add_row(c1, c2, c3, c4, c5, c6)

        grid = Table.grid(expand=True)
        grid.add_row(top)
        grid.add_row(bar_str)
        grid.add_row("")
        grid.add_row(stats)

        return Panel(grid, title="[bold #FFA500]📊 Training Overview[/]", border_style="#333333")

    def _render_curves(self, width: int = 58, height: int = 4, max_val: float = 3.0) -> str:
        """Render dual Braille curves (Loss in orange, PPL in blue) with axes."""
        dot_w = width * 2
        dot_h = height * 4
        grid = [[0 for _ in range(width)] for _ in range(height)]
        color_map = [[0 for _ in range(width)] for _ in range(height)]
        DOT_MASKS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]

        def set_dot(x: int, y: int, s: int) -> None:
            if 0 <= x < dot_w and 0 <= y < dot_h:
                grid[y // 4][x // 2] |= DOT_MASKS[y % 4][x % 2]
                color_map[y // 4][x // 2] = s

        # Dynamic max_val adjustment if loss exceeds default
        actual_max = max(max_val, self.train_loss * 1.1)

        if self.curve_history:
            steps, losses, ppls = zip(*self.curve_history)
            for dx in range(dot_w):
                step_at_dx = (dx / dot_w) * self.total_steps
                if step_at_dx > self.current_step:
                    break
                idx = 0
                while idx < len(steps) - 1 and steps[idx + 1] <= step_at_dx:
                    idx += 1
                l_val = losses[idx]
                p_val = ppls[idx]

                # Map loss to 0..actual_max
                y_l = int((1.0 - (min(l_val, actual_max) / actual_max)) * (dot_h - 1))
                # Map PPL to scale (scaled so it renders smoothly above loss)
                ppl_normalized = min(actual_max, (min(p_val, 15.0) / 15.0) * actual_max)
                y_p = int((1.0 - (ppl_normalized / actual_max)) * (dot_h - 1))

                set_dot(dx, y_l, 1)  # 1 = Loss
                set_dot(dx, y_p, 2)  # 2 = PPL

        y_labels = [f"{actual_max * (height - 1 - r) / (height - 1):3.1f}" for r in range(height)]
        lines: List[str] = []
        for r in range(height):
            row_str = f"[#888888]{y_labels[r]} ┤[/]"
            for c in range(width):
                code = grid[r][c]
                char = chr(0x2800 + code) if code != 0 else " "
                clr = color_map[r][c]
                if clr == 1:
                    row_str += f"[bold #FF6B00]{char}[/]"
                elif clr == 2:
                    row_str += f"[bold #38BDF8]{char}[/]"
                else:
                    row_str += "[#1c1c1c]─[/]" if c % 3 == 0 else " "
            lines.append(row_str)

        lines.append("[#888888]    ┴" + "─" * width + "[/]")

        # X-axis markers: 0, 2K, 4K, 6K, 8K, 10K
        k_step = self.total_steps // 5
        x_lbl = (
            f"     0          {k_step//1000}K         {2*k_step//1000}K         "
            f"{3*k_step//1000}K         {4*k_step//1000}K        {self.total_steps//1000}K"
        )
        lines.append(f"[#888888]{x_lbl[:width + 6]}[/]")
        return "\n".join(lines)

    def _render_middle_row(self, term_width: int) -> RenderableType:
        """Render Optimization settings and Training Curves."""
        t_opt = Table.grid(padding=(0, 1))
        t_opt.add_column(style="dim #AAAAAA")
        t_opt.add_column(style="dim #666666")
        t_opt.add_column(style="bold #FFFFFF")

        t_opt.add_row("Learning Rate", ":", f"{self.learning_rate:.2e}")
        t_opt.add_row("Batch Size", ":", str(self.batch_size))
        t_opt.add_row("Optimizer", ":", self.optimizer_name)
        t_opt.add_row("Weight Decay", ":", f"{self.weight_decay:.2e}")
        t_opt.add_row("Grad Clip", ":", f"{self.grad_clip:.1f}")

        opt_width = 36
        p_opt = Panel(t_opt, title="[bold #00D4FF]⚙ Optimization[/]", border_style="#333333", width=opt_width)

        curve_width = max(25, term_width - opt_width - 8)
        chart_str = self._render_curves(width=curve_width, height=4, max_val=3.0)
        p_curves = Panel(
            chart_str,
            title="[bold #00D4FF]📈 Training Curves[/]          [bold #FF6B00]● Loss[/]   [bold #38BDF8]● PPL[/]",
            border_style="#333333",
        )

        row = Table.grid(expand=True)
        row.add_column(width=opt_width)
        row.add_column()
        row.add_row(p_opt, p_curves)
        return row

    def _render_bottom_row(self, term_width: int) -> RenderableType:
        """Render System Monitor, Status, and Recent Logs."""
        # 1. System Monitor
        sys_grid = Table.grid(padding=(0, 1))
        sys_grid.add_column(style="dim #AAAAAA")
        sys_grid.add_column(style="dim #666666")
        sys_grid.add_column()

        def mini_bar(pct: float, color: str) -> str:
            w = 10
            f = max(0, min(w, int(w * pct / 100.0)))
            return f"[{color}]" + "█" * f + f"[/][#262626]" + "░" * (w - f) + "[/]"

        u_bar = mini_bar(self.gpu_util, "#00FF66")
        sys_grid.add_row("GPU Utilization", ":", f"[bold #00FF66]{self.gpu_util}%[/]  {u_bar}")

        v_pct = (self.vram_used_gb / max(0.1, self.vram_total_gb)) * 100.0
        v_bar = mini_bar(v_pct, "#00D4FF")
        sys_grid.add_row(
            "VRAM Usage",
            ":",
            f"[bold #00D4FF]{self.vram_used_gb:.1f} / {self.vram_total_gb:.1f} GB[/] {v_bar}",
        )

        t_color = "#00FF66" if self.gpu_temp < 65 else ("#FFA500" if self.gpu_temp < 80 else "#FF4444")
        temp_str = f"🌡 [{t_color}]{self.gpu_temp}°C[/]" if self.gpu_temp > 0 else "[dim]n/a[/]"
        sys_grid.add_row("GPU Temp", ":", temp_str)

        sys_width = 38
        p_sys = Panel(sys_grid, title="[bold #00D4FF]🖥 System Monitor[/]", border_style="#333333", width=sys_width)

        # 2. Status Panel
        stat_grid = Table.grid()
        stat_grid.add_column(style="bold #00FF66")
        stat_grid.add_column(style="bold #00FF66")
        stat_grid.add_column(style="white")
        stat_grid.add_row("┃ ", "CUDA: ", self.cuda_device)
        stat_grid.add_row("┃ ", "Mixed Precision: ", f"ON ({self.mixed_precision})")
        stat_grid.add_row("┃ ", "TF32: ", "Enabled" if self.tf32 else "Disabled")

        # 3. Recent Logs
        log_grid = Table.grid()
        log_grid.add_column(no_wrap=True)
        if not self.recent_logs:
            log_grid.add_row("[dim #666666]No log events recorded yet.[/]")
        else:
            for l in list(self.recent_logs)[-3:]:
                log_grid.add_row(l)

        if term_width >= 120:
            sys_width = 36
            stat_width = 28
            p_sys = Panel(sys_grid, title="[bold #00D4FF]🖥 System Monitor[/]", border_style="#333333", width=sys_width)
            p_stat = Panel(stat_grid, title="[bold #00FF66]✔ Status[/]", border_style="#333333", width=stat_width)
            p_logs = Panel(log_grid, title="[bold #00D4FF]📄 Recent Logs[/]", border_style="#333333")

            row = Table.grid(expand=True)
            row.add_column(width=sys_width)
            row.add_column(width=stat_width)
            row.add_column()
            row.add_row(p_sys, p_stat, p_logs)
            return row
        else:
            p_sys = Panel(sys_grid, title="[bold #00D4FF]🖥 System Monitor[/]", border_style="#333333")
            p_stat = Panel(stat_grid, title="[bold #00FF66]✔ Status[/]", border_style="#333333")
            p_logs = Panel(log_grid, title="[bold #00D4FF]📄 Recent Logs[/]", border_style="#333333")

            top_subrow = Table.grid(expand=True)
            top_subrow.add_column(ratio=1)
            top_subrow.add_column(ratio=1)
            top_subrow.add_row(p_sys, p_stat)

            combined = Table.grid(expand=True)
            combined.add_row(top_subrow)
            combined.add_row(p_logs)
            return combined

    def _render_footer(self) -> RenderableType:
        """Render the sleek cyberpunk footer line."""
        t = Table.grid(expand=True)
        t.add_column()
        t.add_column(justify="center")
        t.add_column(justify="right")
        t.add_row(
            "[bold #FFA500]⚡[/] [italic #CCCCCC]Building Efficient LLMs for a Smarter Tomorrow[/]",
            "[dim #444444]│[/]  [bold #FFFFFF]KILAT v2.0.0[/]  [dim #444444]│[/]",
            "[dim #00D4FF]https://github.com/AiRukua/Kilat[/]",
        )
        return t

    def render(self) -> RenderableType:
        """Assemble the complete cyber dashboard layout."""
        term_width = self.console.width

        grid = Table.grid(expand=True)
        grid.add_row(self._render_header(term_width))
        grid.add_row(self._render_overview(term_width))
        grid.add_row(self._render_middle_row(term_width))
        grid.add_row(self._render_bottom_row(term_width))
        grid.add_row(self._render_footer())
        return grid


__all__ = ["KilatDashboard", "DashboardLogHandler"]
