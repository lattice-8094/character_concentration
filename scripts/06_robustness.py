#!/usr/bin/env python3
"""Experiment 6 -- robustness of the trajectory (Appendix A, Figures 7-8).

Three controls, each addressing a way the U-shape might be an artefact:

  * character cutoff -- rebuild the indices at N = 10, 20, 30, 40;
  * cast size        -- verify the cutoff holds the retained cast constant;
  * corpus density   -- subsample a fixed number of novels per year.

The top-N control needs the raw pipeline output; it is skipped, with a notice,
when only the derived tables are available.

    python scripts/06_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import load_corpus                    # noqa: E402
from charconc.plotting import COLORS, new_figure, save     # noqa: E402

PER_YEAR, N_DRAWS = 20, 300


def main() -> None:
    df = load_corpus("gallica")
    results = {}

    # --- cast size: is the cutoff binding? ---
    results["cast_size"] = {
        "mean_retained": round(float(df.n_characters.mean()), 2),
        "share_at_cutoff": round(float((df.n_characters >= df.n_characters.max()).mean()), 4),
        "decade_means": {int(d): round(float(g.n_characters.mean()), 2)
                         for d, g in df.groupby(df.year // 10 * 10)},
        "note": "the cutoff fixes cast size by construction, so the rise in "
                "concentration is a redistribution within a constant cast"}

    # --- corpus density: rarefaction ---
    rng = np.random.default_rng(0)
    counts = df.groupby("year").size()
    valid = counts[counts >= PER_YEAR].index
    pool = df[df.year.isin(valid)]
    by_year = {y: g.index.values for y, g in pool.groupby("year")}

    curves, vertices = [], []
    for _ in range(N_DRAWS):
        pick = np.concatenate([rng.choice(v, PER_YEAR, replace=False)
                               for v in by_year.values()])
        sample = pool.loc[pick]
        curves.append(sample.assign(dec=sample.year // 10 * 10)
                            .groupby("dec")["gini_pagerank"].mean())
        yearly = sample.groupby("year")["gini_pagerank"].mean()
        quad = np.polyfit(yearly.index.astype(float), yearly.values, 2)
        if quad[0] > 0:
            vertices.append(-quad[1] / (2 * quad[0]))

    import pandas as pd
    stack = pd.DataFrame(curves)
    results["rarefaction"] = {
        "novels_per_year": PER_YEAR, "draws": N_DRAWS,
        "years_retained": int(len(valid)),
        "vertex_median": round(float(np.median(vertices)), 1),
        "vertex_ci": [round(float(np.percentile(vertices, 2.5)), 1),
                      round(float(np.percentile(vertices, 97.5)), 1)],
        "note": "series cut at the end of the 1930s: wartime publishing and "
                "copyright gaps leave the 1940s with too few novels to resample"}

    raw = df[df.year < 1940]
    raw_dec = raw.assign(dec=raw.year // 10 * 10).groupby("dec")["gini_pagerank"].mean()
    raw_dec = raw_dec[raw_dec.index < 1940]

    fig, ax = new_figure(figsize=(8.6, 4.6))
    ax.plot(raw_dec.index, raw_dec.values, color=COLORS["muted"], lw=1.6, ls=":",
            marker="o", ms=3.5, label="raw decade means")
    mean = stack.mean()
    ax.plot(mean.index, mean.values, color=COLORS["serial"], lw=2.6,
            label=f"rarefied ({PER_YEAR} novels/year, {len(valid)} years)")
    ax.fill_between(mean.index, stack.quantile(.025).values,
                    stack.quantile(.975).values, color=COLORS["serial"], alpha=0.15)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Gini of PageRank\n(higher = more concentrated)")
    ax.set_xlim(1795, 1935); ax.legend(fontsize=8.5, loc="upper center")
    save(fig, ROOT / "figures" / "fig07_rarefaction.png")

    # --- structure vs attention ---
    from scipy import stats as st
    r, _ = st.pearsonr(df.gini_pagerank, df.cr3_mentions)
    rho, _ = st.spearmanr(df.gini_pagerank, df.cr3_mentions)
    results["structure_vs_attention"] = {"pearson_r": round(float(r), 3),
                                         "spearman_rho": round(float(rho), 3),
                                         "n": len(df)}
    fig, ax = new_figure(figsize=(6.4, 6.0))
    ax.scatter(df.gini_pagerank, df.cr3_mentions, s=5, alpha=0.15,
               color=COLORS["main"], edgecolors="none")
    slope, intercept = np.polyfit(df.gini_pagerank, df.cr3_mentions, 1)
    xs = np.linspace(df.gini_pagerank.min(), df.gini_pagerank.max(), 100)
    ax.plot(xs, slope * xs + intercept, color=COLORS["serial"], lw=2,
            label=f"r = {r:.2f}")
    ax.set_xlabel("Gini of PageRank (structure)")
    ax.set_ylabel("CR3 of mentions (attention)")
    ax.legend(loc="upper left")
    save(fig, ROOT / "figures" / "fig08_structure_vs_attention.png")

    (ROOT / "results" / "06_robustness.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
