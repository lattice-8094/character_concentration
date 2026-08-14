"""Inference helpers: author-clustered OLS, permutation, bootstrap.

The serialization label is a property of the *author*, so novels by the same
writer are not independent observations. Every estimate in the paper is
therefore either clustered by author, aggregated to the author level, or
tested by permuting labels across authors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["design_matrix", "clustered_ols", "author_permutation_test",
           "author_bootstrap_curve", "author_level_test"]


def design_matrix(df: pd.DataFrame, treatment: str | None = None,
                  controls=("detective", "adventure", "canon"),
                  period_ref: str | None = None) -> pd.DataFrame:
    """Build the regression design: controls + period dummies (+ treatment)."""
    dummies = pd.get_dummies(df["period"], prefix="", prefix_sep="")
    if period_ref and period_ref in dummies:
        dummies = dummies.drop(columns=[period_ref])
    else:
        dummies = dummies.iloc[:, 1:]
    cols = [c for c in controls if c in df]
    if treatment:
        cols = cols + [treatment]
    X = pd.concat([df[cols].astype(float), dummies.astype(float)], axis=1)
    return sm.add_constant(X)


def clustered_ols(df: pd.DataFrame, outcome: str = "gini_pagerank",
                  treatment: str | None = None, cluster: str = "author",
                  **kwargs):
    """OLS with standard errors clustered on `cluster` (default: author)."""
    X = design_matrix(df, treatment=treatment, **kwargs)
    y = df[outcome].astype(float)
    return sm.OLS(y, X).fit(cov_type="cluster",
                            cov_kwds={"groups": df[cluster]})


def author_level_test(df: pd.DataFrame, group: str = "serial",
                      outcome: str = "gini_pagerank"):
    """Aggregate to one observation per author, then compare the two groups."""
    from scipy import stats
    agg = (df.groupby("author")
             .agg(value=(outcome, "mean"), group=(group, "first"),
                  n_novels=(outcome, "size"))
             .reset_index())
    a = agg[agg.group == 1].value
    b = agg[agg.group == 0].value
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return {
        "n_authors_treated": len(a), "n_authors_control": len(b),
        "mean_treated": a.mean(), "mean_control": b.mean(),
        "cohens_d": (a.mean() - b.mean()) / pooled,
        "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
        "mannwhitney_p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
    }


def author_permutation_test(df: pd.DataFrame, treatment: str = "serial",
                            outcome: str = "gini_pagerank",
                            n_draws: int = 100_000, seed: int = 12345):
    """Permute the label across authors, holding the group size fixed.

    Returns the two-sided p-value with the (b+1)/(B+1) correction.
    """
    rng = np.random.default_rng(seed)
    labels = df.groupby("author")[treatment].first()
    authors = labels.index.values
    n_treated = int(labels.sum())

    base = design_matrix(df, treatment=None)
    codes = pd.Categorical(df.author, categories=authors).codes
    y = df[outcome].values.astype(float)
    B0 = base.values.astype(float)

    def coef(col):
        X = np.column_stack([B0, col])
        return np.linalg.lstsq(X, y, rcond=None)[0][-1]

    observed = coef(pd.Categorical(df.author, categories=authors)
                    .map(labels).astype(float).to_numpy()
                    if hasattr(pd.Categorical(df.author), "map")
                    else labels.values[codes])
    count = 0
    for _ in range(n_draws):
        perm = np.zeros(len(authors))
        perm[rng.choice(len(authors), n_treated, replace=False)] = 1
        if abs(coef(perm[codes])) >= abs(observed):
            count += 1
    return {"observed": observed, "n_draws": n_draws,
            "p_value": (count + 1) / (n_draws + 1)}


def author_bootstrap_curve(df: pd.DataFrame, years, window: int = 15,
                           outcome: str = "gini_pagerank",
                           n_draws: int = 400, seed: int = 0):
    """Rolling-mean trajectory with uncertainty from resampling *authors*.

    Resampling authors rather than novels keeps the uncertainty consistent
    with the clustered inference used elsewhere.
    """
    rng = np.random.default_rng(seed)
    authors = df.author.unique()
    span = np.arange(min(years) - 5, max(years) + 6)
    curves = []
    for _ in range(n_draws):
        sample = rng.choice(authors, len(authors), replace=True)
        sub = pd.concat([df[df.author == a] for a in sample])
        series = (sub.groupby("year")[outcome].mean().reindex(span)
                     .rolling(window, center=True, min_periods=5).mean())
        curves.append(series.reindex(years).values)
    stack = np.vstack(curves)
    return (np.nanmean(stack, axis=0), np.nanstd(stack, axis=0))
