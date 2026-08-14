"""Character-network construction from Propp-fr / BookNLP-style output.

Pipeline (Section 3.1 of the paper):

    .book + .entities  ->  character ranking  ->  top-N cast
                       ->  sentence-level co-occurrence  ->  weighted graph
                       ->  PageRank  ->  concentration indices

The `.book` files in the CHAPITRES release are Python ``repr(dict)`` dumps,
one dict per line, without an outer list and with single quotes. `load_book`
handles that format alongside plain JSON.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from .metrics import concentration_ratio, gini, normalized_entropy

__all__ = ["load_book", "character_table", "choose_cast", "build_network",
           "network_indices"]

DEFAULT_TOP_N = 20


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _coerce(line: str):
    """Parse one record, tolerating JSON, Python repr, and bare nan/inf."""
    try:
        return json.loads(line)
    except Exception:
        pass
    safe = line.replace("nan", "None").replace("-inf", "None").replace("inf", "None")
    return ast.literal_eval(safe)


def load_book(path: str | Path) -> list[dict]:
    """Load a .book file into a list of character records.

    Tries, in order: whole-file JSON, whole-file literal_eval, then a
    line-by-line pass splitting on ``}\\n{``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for parser in (json.loads, lambda t: ast.literal_eval(
            t.replace("nan", "None").replace("inf", "None"))):
        try:
            obj = parser(text)
            if isinstance(obj, dict):
                obj = obj.get("characters", [obj])
            return list(obj)
        except Exception:
            continue

    records, buf = [], []
    for raw in text.splitlines():
        buf.append(raw)
        if raw.rstrip().endswith("}"):
            chunk = "\n".join(buf).strip()
            buf = []
            if not chunk:
                continue
            try:
                records.append(_coerce(chunk))
            except Exception:
                continue
    if not records:
        raise ValueError(f"could not parse {path}")
    return records


def character_table(doc_id: str, records: list[dict]) -> pd.DataFrame:
    """Flatten character records into a tidy table."""
    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        names = rec.get("names") or rec.get("mentions") or []
        best = ""
        if isinstance(names, list) and names:
            first = names[0]
            best = first.get("n", "") if isinstance(first, dict) else str(first)
        rows.append({
            "doc_id": doc_id,
            "char_id": rec.get("id", rec.get("char_id")),
            "char_name": rec.get("name", best),
            "mention_count": rec.get("count", rec.get("mention_count", 0)),
            "is_singular": bool(rec.get("g", rec.get("is_singular", True))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# cast selection and graph
# --------------------------------------------------------------------------
def choose_cast(table: pd.DataFrame, top_n: int = DEFAULT_TOP_N):
    """Return the top-N singular characters by mention count.

    The cutoff is what fixes cast size in the analysis: 98% of Gallica novels
    reach it exactly, so the concentration indices describe how centrality is
    shared *among* a constant number of retained vertices.
    """
    tbl = table[table.is_singular] if "is_singular" in table else table
    tbl = tbl.sort_values("mention_count", ascending=False).head(top_n)
    return (list(tbl.char_id),
            dict(zip(tbl.char_id, tbl.char_name)),
            dict(zip(tbl.char_id, tbl.mention_count)))


def build_network(entities_path: str | Path, cast_ids, name_map,
                  window: str = "sentence") -> nx.Graph:
    """Weighted co-occurrence graph over the retained cast.

    Edge weight = number of windows (default: sentences) in which both
    characters are mentioned.
    """
    ents = pd.read_csv(entities_path, sep="\t")
    key = {"sentence": "sentence_ID", "paragraph": "paragraph_ID"}[window]
    ents = ents[ents.COREF.isin(cast_ids)]

    graph = nx.Graph()
    for cid in cast_ids:
        graph.add_node(cid, name=name_map.get(cid, str(cid)))

    for _, group in ents.groupby(key):
        present = sorted(set(group.COREF))
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def network_indices(graph: nx.Graph, mention_map: dict) -> dict:
    """Compute the attention and structure indices for one novel."""
    pr = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else \
        {n: 1 / max(graph.number_of_nodes(), 1) for n in graph}
    pr_vals = list(pr.values())
    mentions = [mention_map.get(n, 0) for n in graph.nodes]
    weights = [d["weight"] for *_, d in graph.edges(data=True)] or [0]

    return {
        "n_characters": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        # attention layer
        "cr1_mentions": concentration_ratio(mentions, 1),
        "cr3_mentions": concentration_ratio(mentions, 3),
        "gini_mentions": gini(mentions),
        # structure layer
        "gini_pagerank": gini(pr_vals),
        "cr3_pagerank": concentration_ratio(pr_vals, 3),
        "entropy_pagerank": normalized_entropy(pr_vals),
        "gini_edges": gini(weights),
    }
