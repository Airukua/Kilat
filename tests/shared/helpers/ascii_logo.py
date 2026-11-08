"""
ASCII Logo banner for KILAT, inspired by modern Claude/Anthropic terminal aesthetic.
"""

def get_kilat_banner(version: str = "2.0.0") -> str:
    # ANSI escape sequences for vibrant styling
    AMBER = "\033[38;2;245;158;11m"
    ORANGE = "\033[38;2;234;88;12m"
    YELLOW = "\033[38;2;252;211;77m"
    CYAN = "\033[38;2;6;182;212m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"

    banner = f"""
{ORANGE}  ██╗  ██╗██╗██╗      █████╗ ████████╗{RESET}
{ORANGE}  ██║ ██╔╝██║██║     ██╔══██╗╚══██╔══╝{RESET}
{AMBER}  █████╔╝ ██║██║     ███████║   ██║   {RESET}
{AMBER}  ██╔═██╗ ██║██║     ██╔══██║   ██║   {RESET}
{YELLOW}  ██║  ██╗██║███████╗██║  ██║   ██║   {RESET}
{YELLOW}  ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   {RESET}
{DIM}  ────────────────────────────────────────────────────────────{RESET}
  {BOLD}{CYAN}⚡ KILAT{RESET} {DIM}|{RESET} {ITALIC}Kernelized Lightweight Attention Transformer{RESET} {DIM}(v{version}){RESET}
  {DIM}Architecture: Hybrid Sub-Quadratic Global Decay + Latent MLA{RESET}
{DIM}  ────────────────────────────────────────────────────────────{RESET}
"""
    return banner


def print_kilat_banner(version: str = "2.0.0") -> None:
    print(get_kilat_banner(version))


if __name__ == "__main__":
    print_kilat_banner()
