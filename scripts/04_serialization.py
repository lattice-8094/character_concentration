#!/usr/bin/env python3
"""Experiment 4 -- the serialization test (Section 5.6, Figure 6).

The headline result. Because the label is a property of the *author*, novels
by the same writer are not independent observations, so the effect is reported
three ways that do not assume independence:

  (a) author-level means, one observation per writer;
  (b) OLS net of period, genre and canon, errors clustered by author;
  (c) a permutation test reassigning the label across authors.

It also tests two candidate explanations for the post-1880 rise inside the
serial group: cast size (held fixed by the cutoff) and the recurring hero.

    python scripts/04_serialization.py [--draws 100000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import (attach_labels, fold, load_author_labels,        # noqa: E402
                             load_corpus)
from charconc.inference import (author_bootstrap_curve, author_level_test,   # noqa: E402
                                author_permutation_test, clustered_ols)
from charconc.plotting import COLORS, save                                   # noqa: E402
import matplotlib.pyplot as plt                                              # noqa: E402

AGE = (1840, 1914)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=100_000,
                    help="permutation draws (use 5000 for a quick run)")
    args = ap.parse_args()

    labels = load_author_labels()
    df = attach_labels(load_corpus("gallica"), labels)
    panel = df.dropna(subset=["serial"]).copy()
    panel["serial"] = panel.serial.astype(int)

    results = {"n_novels": len(panel),
               "n_authors": int(panel.author.nunique()),
               "n_serial_novels": int((panel.serial == 1).sum()),
               "n_control_novels": int((panel.serial == 0).sum()),
               "n_serial_authors": int(panel[panel.serial == 1].author.nunique()),
               "n_control_authors": int(panel[panel.serial == 0].author.nunique())}

    # (a) author level
    results["author_level"] = {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in author_level_test(panel).items()}

    # (b) clustered OLS
    model = clustered_ols(panel, treatment="serial")
    ci = model.conf_int()
    results["clustered_ols"] = {
        "coef": round(float(model.params["serial"]), 4),
        "ci_low": round(float(ci.loc["serial", 0]), 4),
        "ci_high": round(float(ci.loc["serial", 1]), 4),
        "p_value": float(model.pvalues["serial"])}

    # (c) permutation across authors
    results["permutation"] = author_permutation_test(panel, n_draws=args.draws)
    results["permutation"]["observed"] = round(
        float(results["permutation"]["observed"]), 4)

    # cast held strictly constant
    at_cap = panel[panel.n_characters >= panel.n_characters.max()]
    results["cast_at_cutoff"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in author_level_test(at_cap).items()}
    results["cast_at_cutoff"]["n_novels"] = len(at_cap)

    # internal rise after 1880
    serial = panel[panel.serial == 1]
    for label, lo, hi in [("before_1880", 1840, 1880), ("after_1880", 1880, 1914)]:
        window = serial[serial.year.between(lo, hi)]
        X = sm.add_constant(window[["year"]].astype(float))
        fit = sm.OLS(window.gini_pagerank.astype(float), X).fit(
            cov_type="cluster", cov_kwds={"groups": window.author})
        entry = {"slope": round(float(fit.params["year"]), 5),
                 "p_clustered": float(fit.pvalues["year"]),
                 "n_novels": len(window)}
        if label == "after_1880":       # add author fixed effects
            fe = pd.get_dummies(window.author, prefix="a", drop_first=True).astype(float)
            fit_fe = sm.OLS(window.gini_pagerank.astype(float),
                            pd.concat([X, fe], axis=1)).fit(
                cov_type="cluster", cov_kwds={"groups": window.author})
            entry["slope_with_author_fe"] = round(float(fit_fe.params["year"]), 5)
            entry["p_with_author_fe"] = float(fit_fe.pvalues["year"])
        results[f"internal_{label}"] = entry

    # does the serial group rise *faster* after 1880? test the interaction
    # directly rather than comparing two separately estimated slopes
    window = panel[panel.year.between(1880, 1914)].copy()
    window["yr_c"] = window.year - 1880
    X = sm.add_constant(pd.DataFrame({
        "yr_c": window.yr_c.astype(float),
        "serial": window.serial.astype(float),
        "serial_x_yr": (window.serial * window.yr_c).astype(float)}))
    inter = sm.OLS(window.gini_pagerank.astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": window.author})
    results["slope_interaction"] = {
        "slope_control": round(float(inter.params["yr_c"]), 5),
        "slope_serial": round(float(inter.params["yr_c"] + inter.params["serial_x_yr"]), 5),
        "interaction": round(float(inter.params["serial_x_yr"]), 5),
        "p_value": float(inter.pvalues["serial_x_yr"]),
        "note": "the two slopes are not statistically distinguishable; we claim "
                "only that concentration also rises within the serial group"}

    # recurring hero -- tested at author level, where n is small
    heroes = [fold(a) for a in labels["recurring_hero"]["authors"]]
    serial = serial.copy()
    serial["hero"] = serial.author.isin(heroes).astype(int)
    agg = serial.groupby("author").agg(value=("gini_pagerank", "mean"),
                                       hero=("hero", "first")).reset_index()
    a, b = agg[agg.hero == 1].value, agg[agg.hero == 0].value
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    results["recurring_hero"] = {
        "n_authors_with_series": len(a), "n_authors_without": len(b),
        "mean_with_series": round(float(a.mean()), 4),
        "mean_without": round(float(b.mean()), 4),
        "cohens_d": round(float((a.mean() - b.mean()) / pooled), 3),
        "exact_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
        "note": "Tested inside the predefined serialization-oriented group, so "
                "the population matches the main test. Sensitivity: admitting "
                "the four series authors excluded from that group as "
                "publication-ambiguous (Ponson du Terrail, Leblanc, Leroux, "
                "Sazie) gives 11 vs 19 authors, d = -0.39, p = 0.09 -- a "
                "smaller gap in the same direction."}

    # sensitivity: admit the series authors excluded from the main group
    all_heroes = df[df.author.isin(heroes)]
    no_series = df[df.author.isin([a for a in serial.author.unique()
                                   if a not in heroes])]
    both = pd.concat([all_heroes.assign(h=1), no_series.assign(h=0)])
    agg_b = both.groupby("author").agg(value=("gini_pagerank", "mean"),
                                       h=("h", "first")).reset_index()
    x, y = agg_b[agg_b.h == 1].value, agg_b[agg_b.h == 0].value
    pooled_b = np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)
    results["recurring_hero_sensitivity"] = {
        "n_authors_with_series": len(x), "n_authors_without": len(y),
        "cohens_d": round(float((x.mean() - y.mean()) / pooled_b), 3),
        "exact_p": float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)}

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    data = [panel[panel.serial == 1].gini_pagerank,
            panel[panel.serial == 0].gini_pagerank]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, colour in zip(parts["bodies"], [COLORS["serial"], COLORS["control"]]):
        body.set_facecolor(colour); body.set_alpha(0.55)
    parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"feuilleton authors\n({results['n_serial_authors']} authors, "
         f"n={results['n_serial_novels']})",
         f"literary control\n({results['n_control_authors']} authors, "
         f"n={results['n_control_novels']})"], fontsize=9)
    ax.set_ylabel("Gini of PageRank")
    ax.set_title("Distribution of concentration", fontsize=10)

    ax = axes[1]
    years = np.arange(*AGE)
    for value, colour, label in [(1, COLORS["serial"], "feuilleton authors"),
                                 (0, COLORS["control"], "literary control")]:
        group = panel[panel.serial == value]
        mean, se = author_bootstrap_curve(group, years)
        ax.plot(years, mean, color=colour, lw=2.4, label=label)
        ax.fill_between(years, mean - se, mean + se, color=colour, alpha=0.18)
    ax.set_xlim(*AGE); ax.set_ylim(0.34, 0.50)
    ax.set_xlabel("Publication year"); ax.set_ylabel("Gini of PageRank")
    ax.set_title("15-year rolling mean ± 1 SE (bootstrap over authors)", fontsize=9.5)
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, ROOT / "figures" / "fig06_serialization.png")

    (ROOT / "results" / "04_serialization.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items()
                      if k.startswith(("author_level", "clustered", "permutation"))},
                     indent=2))


if __name__ == "__main__":
    main()
