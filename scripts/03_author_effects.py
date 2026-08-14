#!/usr/bin/env python3
"""Experiment 3 -- author signatures and author fixed effects (Section 5.5, Figure 5).

Two questions:
  (a) how much of the variance in concentration is explained by author identity,
      and how sensitive is that share to the inclusion threshold;
  (b) does the U-shape survive author fixed effects, i.e. when each writer is
      compared only to himself over time.

    python scripts/03_author_effects.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import attach_labels, load_author_labels, load_corpus  # noqa: E402
from charconc.inference import design_matrix                                # noqa: E402
from charconc.plotting import new_figure, save                              # noqa: E402

THRESHOLDS = (1, 2, 3, 5, 10, 20)
N_SHOWN = 15


def main() -> None:
    df = attach_labels(load_corpus("gallica"), load_author_labels())

    # (a) variance explained by author identity, across thresholds
    variance = []
    for thr in THRESHOLDS:
        counts = df.author.value_counts()
        sub = df[df.author.isin(counts[counts >= thr].index)]
        dummies = sm.add_constant(pd.get_dummies(sub.author, drop_first=True).astype(float))
        fit = sm.OLS(sub.gini_pagerank.astype(float), dummies).fit()
        variance.append({"min_novels": thr, "n_novels": len(sub),
                         "n_authors": int(sub.author.nunique()),
                         "r2": round(float(fit.rsquared), 3),
                         "r2_adjusted": round(float(fit.rsquared_adj), 3)})

    # (b) do period coefficients survive author fixed effects?
    counts = df.author.value_counts()
    panel = df[df.author.isin(counts[counts >= 5].index)].copy()
    base = design_matrix(panel, controls=())
    plain = sm.OLS(panel.gini_pagerank.astype(float), base).fit()
    fixed = pd.get_dummies(panel.author, prefix="a", drop_first=True).astype(float)
    with_fe = sm.OLS(panel.gini_pagerank.astype(float),
                     pd.concat([base, fixed], axis=1)).fit()
    period_terms = [c for c in base.columns if c.startswith("[")]
    comparison = {t: {"without_fe": round(float(plain.params[t]), 4),
                      "with_fe": round(float(with_fe.params[t]), 4)}
                  for t in period_terms}

    results = {"variance_by_threshold": variance,
               "period_coefficients": comparison,
               "note": "R2 rises mechanically with the number of author dummies; "
                       "the adjusted R2 is the defensible figure."}
    (ROOT / "results" / "03_author_effects.json").write_text(json.dumps(results, indent=2))

    # figure: per-author distributions for the most represented writers
    top = df.author.value_counts().head(N_SHOWN).index[::-1]
    fig, ax = new_figure(figsize=(7.4, 8.2))
    data = [df[df.author == a].gini_pagerank.values for a in top]
    parts = ax.violinplot(data, vert=False, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#cccccc"); body.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    ax.axvline(df.gini_pagerank.mean(), ls="--", color="grey", alpha=0.7,
               label="corpus mean")
    ax.set_yticks(range(1, len(top) + 1))
    ax.set_yticklabels([a.capitalize() for a in top], fontsize=9)
    ax.set_xlabel("Gini of PageRank")
    ax.legend(fontsize=8)
    save(fig, ROOT / "figures" / "fig05_author_signatures.png")

    print(f"{'threshold':>10} {'authors':>8} {'R2':>7} {'R2 adj':>8}")
    for row in variance:
        print(f"{row['min_novels']:>10} {row['n_authors']:>8} "
              f"{row['r2']:>7.3f} {row['r2_adjusted']:>8.3f}")


if __name__ == "__main__":
    main()
