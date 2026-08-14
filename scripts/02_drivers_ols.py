#!/usr/bin/env python3
"""Experiment 2 -- period, genre and canon as joint drivers (Section 5.4, Figure 4).

Regresses concentration on publication period, genre indicators and canonical
status simultaneously, with standard errors clustered by author. Period is
coded in twenty-year bins with 1860-1880 as the reference: it is the modal bin
(which tightens every other interval) and the one containing the trough (which
makes the coefficients read as deviations from the mid-century minimum).

    python scripts/02_drivers_ols.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import attach_labels, load_author_labels, load_corpus  # noqa: E402
from charconc.inference import clustered_ols                                # noqa: E402
from charconc.plotting import COLORS, new_figure, save                      # noqa: E402

PERIOD_REF = "[1860, 1880)"


def main() -> None:
    df = attach_labels(load_corpus("gallica"), load_author_labels())
    df = df[df.is_canon.notna()].copy()

    model = clustered_ols(df, period_ref=PERIOD_REF)
    ci = model.conf_int()

    terms = [c for c in model.params.index
             if c not in ("const", "detective", "adventure", "canon")]
    terms += ["detective", "adventure", "canon"]
    table = {t: {"coef": round(float(model.params[t]), 4),
                 "ci_low": round(float(ci.loc[t, 0]), 4),
                 "ci_high": round(float(ci.loc[t, 1]), 4),
                 "p_clustered": float(model.pvalues[t])} for t in terms}
    results = {"n_novels": int(model.nobs), "n_authors": int(df.author.nunique()),
               "r_squared": round(float(model.rsquared), 4),
               "period_reference": PERIOD_REF, "coefficients": table}

    fig, ax = new_figure(figsize=(6.8, 4.8))
    ypos = np.arange(len(terms))[::-1]
    for y, term in zip(ypos, terms):
        colour = (COLORS["serial"] if term in ("detective", "adventure")
                  else COLORS["canon"] if term == "canon" else COLORS["main"])
        ax.errorbar(model.params[term], y,
                    xerr=[[model.params[term] - ci.loc[term, 0]],
                          [ci.loc[term, 1] - model.params[term]]],
                    fmt="o", color=colour, capsize=3, ms=5)
    ax.axvline(0, ls="--", color="grey", alpha=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t.replace("[", "").replace(")", "").replace(", ", "–")
                        .replace("detective", "detective fiction")
                        .replace("adventure", "adventure fiction")
                        .replace("canon", "canonical") for t in terms], fontsize=9)
    ax.set_xlabel("Coefficient (net effect on Gini of PageRank)")
    ax.set_title(f"Drivers of concentration (n={int(model.nobs):,}, "
                 f"R²={model.rsquared:.3f})\nreference 1860–1880; "
                 "errors clustered by author", fontsize=9.5)
    ax.grid(axis="x")
    save(fig, ROOT / "figures" / "fig04_drivers_ols.png")

    (ROOT / "results" / "02_drivers_ols.json").write_text(json.dumps(results, indent=2))
    for term, stats_ in table.items():
        flag = "" if stats_["p_clustered"] < 0.05 else "   (n.s.)"
        print(f"  {term:16s} {stats_['coef']:+.4f}  p={stats_['p_clustered']:.4f}{flag}")


if __name__ == "__main__":
    main()
