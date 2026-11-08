"""Shared ASCII CLI banner utility for Kilat."""

from __future__ import annotations


def get_kilat_banner(version: str = "2.0.0") -> str:
    """Return the styled KILAT terminal banner string."""
    amber = "\033[38;2;245;158;11m"
    orange = "\033[38;2;234;88;12m"
    yellow = "\033[38;2;252;211;77m"
    cyan = "\033[38;2;6;182;212m"
    dim = "\033[2m"
    bold = "\033[1m"
    italic = "\033[3m"
    reset = "\033[0m"

    return (
        f"\n"
        f"{orange}  ██╗  ██╗██╗██╗      █████╗ ████████╗{reset}\n"
        f"{orange}  ██║ ██╔╝██║██║     ██╔══██╗╚══██╔══╝{reset}\n"
        f"{amber}  █████╔╝ ██║██║     ███████║   ██║   {reset}\n"
        f"{amber}  ██╔═██╗ ██║██║     ██╔══██║   ██║   {reset}\n"
        f"{yellow}  ██║  ██╗██║███████╗██║  ██║   ██║   {reset}\n"
        f"{yellow}  ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   {reset}\n"
        f"{dim}  ────────────────────────────────────────────────────────────{reset}\n"
        f"  {bold}{cyan}⚡ KILAT{reset} {dim}|{reset} {italic}Kernelized Lightweight Attention Transformer{reset} {dim}(v{version}){reset}\n"
        f"  {dim}Architecture: Hybrid Sub-Quadratic Global Decay + Latent MLA{reset}\n"
        f"{dim}  ────────────────────────────────────────────────────────────{reset}\n"
    )


def print_kilat_banner(version: str = "2.0.0", quiet: bool = False) -> None:
    """Print the KILAT banner unless ``quiet=True``."""
    if not quiet:
        print(get_kilat_banner(version=version))


__all__ = ["get_kilat_banner", "print_kilat_banner"]
