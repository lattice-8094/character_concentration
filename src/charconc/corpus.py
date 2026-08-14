"""Corpus loading, author normalization, and the serialization labels."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

__all__ = ["fold", "author_of", "load_corpus", "load_author_labels",
           "attach_labels", "WINDOW"]

#: Analysis window used throughout the paper.
WINDOW = (1800, 1940)

DATA = Path(__file__).resolve().parents[2] / "data"


def fold(text) -> str:
    """Accent-fold and lowercase, for matching author names across sources."""
    text = unicodedata.normalize("NFKD", str(text))
    return text.encode("ascii", "ignore").decode().lower()


#: Surname particles that must stay attached to the following token, so that
#: "Le Roux", "Du Camp" or "La Jeunesse" do not collapse to "le", "du", "la".
PARTICLES = {"le", "la", "les", "du", "de", "des", "d", "van", "von", "della"}


def author_of(file_name: str) -> str:
    """Extract a normalized author surname from a corpus filename.

    Filenames are ``YEAR_Author_Title...``. Gallica separates surname and given
    name with ``_``; CHAPITRES packs them into one field with ``-``. We fold the
    author field and take its first token, joining a leading particle to the
    token that follows it.

    >>> author_of("1863_Feval_Paul_Les-Habits-noirs")
    'feval'
    >>> author_of("1900_Le-Roux-Hugues_LEnfant-du-mystere")
    'le-roux'
    """
    parts = str(file_name).split("_")
    if len(parts) < 2:
        return ""
    tokens = fold(parts[1]).replace("-", " ").split()
    if not tokens:
        return ""
    if tokens[0] in PARTICLES and len(tokens) > 1:
        return f"{tokens[0]}-{tokens[1]}"
    return tokens[0]


def load_corpus(which: str = "gallica", window: tuple | None = WINDOW) -> pd.DataFrame:
    """Load a derived concentration table with author column attached."""
    path = DATA / "derived" / f"concentration_{which}.csv"
    df = pd.read_csv(path).dropna(subset=["year", "gini_pagerank"])
    df["author"] = df.file_name.map(author_of)
    df = df[df.author != ""]
    if window:
        df = df[(df.year >= window[0]) & (df.year < window[1])]
    return df.reset_index(drop=True)


def load_author_labels(path: str | Path | None = None) -> dict:
    """Load the feuilleton / literary-control author lists."""
    path = Path(path) if path else DATA / "labels" / "authors.yaml"
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def attach_labels(df: pd.DataFrame, labels: dict,
                  feuilleton_groups=("core", "ambiguous", "late"),
                  control_groups=("narrow", "mid_century", "broad")) -> pd.DataFrame:
    """Add a ``serial`` column: 1 = feuilleton author, 0 = control, NaN = unlabelled.

    The groups are kept separate so that the specification curve
    (scripts/05) can switch them on and off.
    """
    feuilleton = [fold(a) for g in feuilleton_groups for a in labels["feuilleton"].get(g, [])]
    control = [fold(a) for g in control_groups for a in labels["control"].get(g, [])]
    out = df.copy()
    out["serial"] = np.where(out.author.isin(feuilleton), 1.0,
                             np.where(out.author.isin(control), 0.0, np.nan))
    for col, src in [("detective", "is_detective"), ("adventure", "is_adventure"),
                     ("canon", "is_canon")]:
        if src in out:
            out[col] = out[src].fillna(False).astype(int)
    out["period"] = pd.cut(out.year, bins=range(WINDOW[0], WINDOW[1] + 1, 20),
                           right=False).astype(str)
    return out
