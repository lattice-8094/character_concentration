#!/usr/bin/env python3
"""Experiment 7 -- concentration across genres (Appendix E).

Gallica carries only two genre labels. The auxiliary corpus has a finer
classification, which lets us check whether detective and adventure fiction
behave idiosyncratically or sit on a wider gradation.

    python scripts/07_genre_gradation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from charconc.corpus import load_corpus       # noqa: E402

MIN_NOVELS = 20


def main() -> None:
    df = load_corpus("chapitres", window=(1800, 1950)).dropna(subset=["genre"])
    overall = df.gini_pagerank.mean()

    table = (df.groupby("genre")
               .agg(n=("gini_pagerank", "size"), mean=("gini_pagerank", "mean"),
                    median_year=("year", "median"))
               .query(f"n >= {MIN_NOVELS}")
               .assign(vs_corpus=lambda t: t["mean"] - overall)
               .sort_values("mean"))

    out = ROOT / "results" / "07_genre_gradation.csv"
    table.round(4).to_csv(out)
    print(f"corpus mean = {overall:.3f}\n")
    print(table.round(3).to_string())
    print(f"\n[results] {out}")


if __name__ == "__main__":
    main()
