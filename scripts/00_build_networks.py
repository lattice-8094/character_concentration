#!/usr/bin/env python3
"""Stage 0 -- build character networks and concentration indices from raw output.

Consumes the Propp-fr output (.book + .entities, one pair per novel) and
produces the derived table every other script reads.

    python scripts/00_build_networks.py --input data/raw/gallica --out data/derived/concentration_gallica.csv

Not required to reproduce the paper: the derived tables are shipped in
data/derived/. Run this only to rebuild them from the raw pipeline output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from charconc.networks import (DEFAULT_TOP_N, build_network, character_table,  # noqa: E402
                               choose_cast, load_book, network_indices)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="directory holding .book and .entities files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", choices=["sentence", "paragraph"], default="sentence")
    args = ap.parse_args()

    rows, skipped = [], []
    books = sorted(args.input.glob("*.book"))
    for i, book in enumerate(books, 1):
        stem = book.stem
        entities = book.with_suffix(".entities")
        if not entities.exists():
            skipped.append((stem, "missing .entities"))
            continue
        try:
            table = character_table(stem, load_book(book))
            cast, names, mentions = choose_cast(table, args.top_n)
            graph = build_network(entities, cast, names, window=args.window)
            row = {"file_name": stem, "year": int(stem.split("_")[0])}
            row.update(network_indices(graph, mentions))
            rows.append(row)
        except Exception as exc:                      # noqa: BLE001
            skipped.append((stem, str(exc)[:120]))
        if i % 500 == 0:
            print(f"  {i}/{len(books)} novels", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[done] {len(rows)} novels -> {args.out}")
    if skipped:
        log = args.out.with_name("skipped_books.csv")
        pd.DataFrame(skipped, columns=["file_name", "reason"]).to_csv(log, index=False)
        print(f"[warn] {len(skipped)} skipped -> {log}")


if __name__ == "__main__":
    main()
