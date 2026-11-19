"""Shared plotting style and 300 DPI export helpers.

All figures use a serif stack (Times New Roman when installed) with STIX
mathematics, a white background and no top/right spines, which reproduces the
layout of the published figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DPI = 300
SERIF_STACK = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]

STYLE = {
    "font.family": "serif",
    "font.serif": SERIF_STACK,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.6,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.unicode_minus": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # embed TrueType subsets in vector output
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Activate the shared style (safe to call once per interpreter)."""
    sns.set_style("ticks")
    plt.rcParams.update(STYLE)


def png_dpi(path: str | Path) -> tuple[float, float] | None:
    """Read the pHYs chunk of a PNG and return (x_dpi, y_dpi)."""
    with open(path, "rb") as fh:
        if fh.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"{Path(path).name} is not a valid PNG file")
        while True:
            length = fh.read(4)
            if len(length) < 4:
                return None
            n = int.from_bytes(length, "big")
            ctype = fh.read(4)
            data = fh.read(n)
            fh.read(4)                                   # CRC
            if ctype == b"pHYs":
                if data[8] != 1:                         # 1 = metres
                    return None
                return (int.from_bytes(data[0:4], "big") * 0.0254,
                        int.from_bytes(data[4:8], "big") * 0.0254)
            if ctype == b"IDAT":
                return None


def save_figure(fig, path: str | Path) -> Path:
    """Write ``fig`` at 300 DPI and verify the embedded resolution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    dpi = png_dpi(path)
    if dpi is None or abs(dpi[0] - DPI) > 0.5 or abs(dpi[1] - DPI) > 0.5:
        raise RuntimeError(f"{path.name}: resolution is not {DPI} DPI ({dpi})")
    print(f"[figure] {path.name}  {path.stat().st_size / 1024:.0f} KB  "
          f"{dpi[0]:.1f} DPI")
    return path
