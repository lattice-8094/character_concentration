"""Concentration indices for character-systems.

Two layers are computed, following the paper:

* **attention**  -- from raw mention counts (CR1, CR3, Gini, HHI)
* **structure**  -- from weighted PageRank over the co-occurrence graph
  (Gini, CR3, normalized entropy)

All indices are computed over the top-N retained characters (N = 20 in the
paper); see `networks.build_network` for how the cast is selected.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gini", "concentration_ratio", "hhi", "normalized_entropy"]


def gini(x) -> float:
    """Gini coefficient of a non-negative distribution.

    0 = every character carries equal weight; -> 1 = one character absorbs
    the whole distribution. Implements the mean-absolute-difference form
    used in the paper (Ceriani & Verme 2012 for the original derivation).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    if np.any(x < 0):
        raise ValueError("Gini is undefined for negative values")
    total = x.sum()
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * total)) - (n + 1) / n)


def concentration_ratio(x, k: int = 3) -> float:
    """Share of the total held by the k largest values (CR-k)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    total = x.sum()
    if total == 0 or x.size == 0:
        return float("nan")
    return float(np.sort(x)[::-1][:k].sum() / total)


def hhi(x) -> float:
    """Herfindahl-Hirschman index of the shares."""
    x = np.asarray(x, dtype=float)
    total = x.sum()
    if total == 0:
        return float("nan")
    return float(np.sum((x / total) ** 2))


def normalized_entropy(x) -> float:
    """Shannon entropy of the shares, normalized to [0, 1].

    Falls as concentration rises, so it moves opposite to the Gini.
    """
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if x.size <= 1:
        return float("nan")
    p = x / x.sum()
    return float(-np.sum(p * np.log(p)) / np.log(x.size))
