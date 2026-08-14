#!/usr/bin/env python3
"""Experiment 1 -- the diachronic trajectory (Section 5.2, Figure 3).

Fits a quadratic to the yearly means, locates the vertex, reports the
trough-to-peak change in interpretable terms, and contrasts it with the
between-novel variance.

    python scripts/01_trajectory.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import attach_labels, load_author_labels, load_corpus  # noqa: E402
from charconc.plotting import COLORS, new_figure, save      # noqa: E402

FEUILLETON_AGE = (1840, 1914)
TROUGH, PEAK = (1855, 1875), (1920, 1940)


def main() -> None:
    df = attach_labels(load_corpus("gallica"), load_author_labels())

    yearly = df.groupby("year")["gini_pagerank"].mean()
    quad = np.polyfit(yearly.index.astype(float), yearly.values, 2)
    vertex = -quad[1] / (2 * quad[0])

    trough = df[df.year.between(*TROUGH)]
    peak = df[df.year.between(*PEAK)]
    pooled = np.sqrt((trough.gini_pagerank.std() ** 2 +
                      peak.gini_pagerank.std() ** 2) / 2)

    # Before/after 1900, clustered by author. Period dummies are omitted here:
    # 1900 is itself a bin boundary, so the split is collinear with them.
    import statsmodels.api as sm
    df["post1900"] = (df.year >= 1900).astype(float)
    model = sm.OLS(df.gini_pagerank.astype(float),
                   sm.add_constant(df[["post1900"]])).fit(
        cov_type="cluster", cov_kwds={"groups": df.author})

    results = {
        "n_novels": len(df),
        "vertex": round(float(vertex), 1),
        "convex": bool(quad[0] > 0),
        "trough_mean": round(float(trough.gini_pagerank.mean()), 4),
        "peak_mean": round(float(peak.gini_pagerank.mean()), 4),
        "trough_to_peak": round(float(peak.gini_pagerank.mean() -
                                      trough.gini_pagerank.mean()), 4),
        "cohens_d": round(float((peak.gini_pagerank.mean() -
                                 trough.gini_pagerank.mean()) / pooled), 3),
        "within_period_sd": round(float(trough.gini_pagerank.std()), 4),
        "share_trough_above_peak_mean": round(
            float((trough.gini_pagerank > peak.gini_pagerank.mean()).mean()), 3),
        "cr1_trough": round(float(trough.cr1_mentions.mean()), 4),
        "cr1_peak": round(float(peak.cr1_mentions.mean()), 4),
        "post1900_coef": round(float(model.params["post1900"]), 4),
        "post1900_p_clustered": float(model.pvalues["post1900"]),
    }

    fig, ax = new_figure()
    roll = yearly.rolling(15, center=True, min_periods=5).mean()
    ax.axvspan(*FEUILLETON_AGE, alpha=0.13, color=COLORS["serial"], zorder=0)
    ax.plot(roll.index, roll.values, color=COLORS["main"], lw=2.4, zorder=3)
    ax.text(sum(FEUILLETON_AGE) / 2, ax.get_ylim()[1] - 0.002,
            "Age of the roman-feuilleton (1840–1914)", ha="center", va="top",
            fontsize=9, color="#a01c1c")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Gini of PageRank\n(higher = more concentrated)")
    ax.set_xlim(1800, 1940)
    save(fig, ROOT / "figures" / "fig03_trajectory.png")

    out = ROOT / "results" / "01_trajectory.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[results] {out}")


if __name__ == "__main__":
    main()
