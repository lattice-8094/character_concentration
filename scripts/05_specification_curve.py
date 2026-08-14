#!/usr/bin/env python3
"""Experiment 5 -- specification curve for the serialization test (Appendix C, Figure 9).

Group composition involves judgement: how broadly to draw the literary control,
whether to keep the authors who worked in both circuits, whether to include the
serial writers of the 1890s-1900s. Rather than report one preferred choice, we
cross the three and estimate all twelve resulting specifications, each with
author-clustered errors.

    python scripts/05_specification_curve.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import attach_labels, load_author_labels, load_corpus  # noqa: E402
from charconc.inference import clustered_ols                                # noqa: E402
from charconc.plotting import COLORS, save                                  # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402

CONTROLS = {"narrow": ("narrow",),
            "+ mid-century": ("narrow", "mid_century"),
            "broad": ("narrow", "mid_century", "broad")}


def main() -> None:
    labels = load_author_labels()
    raw = load_corpus("gallica")

    rows = []
    for (cname, cgroups), ambiguous, late in itertools.product(
            CONTROLS.items(), [True, False], [True, False]):
        fgroups = ["core"] + (["ambiguous"] if ambiguous else []) + \
                  (["late"] if late else [])
        df = attach_labels(raw, labels, feuilleton_groups=fgroups,
                           control_groups=cgroups)
        panel = df.dropna(subset=["serial"]).copy()
        panel["serial"] = panel.serial.astype(int)
        model = clustered_ols(panel, treatment="serial")
        ci = model.conf_int()
        rows.append({"control": cname,
                     "ambiguous": "incl." if ambiguous else "excl.",
                     "late": "incl." if late else "excl.",
                     "coef": float(model.params["serial"]),
                     "ci_low": float(ci.loc["serial", 0]),
                     "ci_high": float(ci.loc["serial", 1]),
                     "p_value": float(model.pvalues["serial"]),
                     "n_serial": int((panel.serial == 1).sum()),
                     "n_control": int((panel.serial == 0).sum()),
                     "n_authors": int(panel.author.nunique())})

    table = pd.DataFrame(rows).sort_values("coef").reset_index(drop=True)
    table.to_csv(ROOT / "results" / "05_specification_curve.csv", index=False)

    summary = {"n_specifications": len(table),
               "all_negative": bool((table.coef < 0).all()),
               "n_significant_5pct": int((table.p_value < 0.05).sum()),
               "coef_min": round(float(table.coef.min()), 4),
               "coef_max": round(float(table.coef.max()), 4),
               "coef_median": round(float(table.coef.median()), 4),
               "p_max": round(float(table.p_value.max()), 4)}
    (ROOT / "results" / "05_specification_curve.json").write_text(
        json.dumps(summary, indent=2))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1]})
    x = np.arange(len(table))
    ax.errorbar(x, table.coef,
                yerr=[table.coef - table.ci_low, table.ci_high - table.coef],
                fmt="o", color=COLORS["serial"], ecolor="#aaaaaa", capsize=3, ms=5)
    ax.axhline(0, ls="--", color="grey", alpha=0.7)
    ax.axhline(table.coef.median(), ls=":", color=COLORS["serial"], alpha=0.6)
    ax.set_ylabel("Feuilleton coefficient\n(clustered by author)", fontsize=9)
    ax.set_title("Specification curve: 12 compositions of the two author groups",
                 fontsize=10)

    options = [("control: broad", table.control == "broad"),
               ("control: + mid-century", table.control == "+ mid-century"),
               ("control: narrow", table.control == "narrow"),
               ("ambiguous authors incl.", table.ambiguous == "incl."),
               ("late serial authors incl.", table.late == "incl.")]
    for i, (label, mask) in enumerate(options):
        y = len(options) - 1 - i
        ax2.scatter(x[mask.values], [y] * mask.sum(), marker="s", s=22,
                    color=COLORS["main"])
        ax2.scatter(x[~mask.values], [y] * (~mask).sum(), marker="s", s=22,
                    color="#e8e8e8")
    ax2.set_yticks(range(len(options)))
    ax2.set_yticklabels([o[0] for o in options][::-1], fontsize=8)
    ax2.set_xticks([])
    ax2.set_xlabel("specifications, ordered by coefficient", fontsize=9)
    save(fig, ROOT / "figures" / "fig09_specification_curve.png")

    print(table[["control", "ambiguous", "late", "coef", "p_value"]].to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
