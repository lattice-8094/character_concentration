"""Shared plotting style so every figure in the paper looks the same."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["COLORS", "new_figure", "save"]

COLORS = {
    "main": "#1f2d5a",      # trajectories, period coefficients
    "serial": "#c0392b",    # feuilleton group, genre coefficients
    "control": "#1f2d5a",   # literary control
    "canon": "#2ca02c",
    "muted": "#888888",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 170,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def new_figure(ncols: int = 1, figsize=(9, 4.8)):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    return fig, axes


def save(fig, path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path)
    print(f"[figure] {path}")
    return path
