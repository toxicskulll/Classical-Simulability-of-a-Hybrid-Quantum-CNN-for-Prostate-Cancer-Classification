"""Shared figure style: vector PDF, fonts >= 8 pt, embedded TrueType fonts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import RESULTS_DIR  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
MIN_FONT_PT = 8.0


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,       # TrueType, editable/searchable text
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save_pdf(fig, path: Path) -> Path:
    """Save as vector PDF after asserting every text element is >= 8 pt."""
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
             if t.get_text() and t.get_fontsize() < MIN_FONT_PT]
    if small:
        raise ValueError(f"text below {MIN_FONT_PT} pt: {small[:5]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path
