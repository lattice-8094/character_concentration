"""Tests for author normalization and label assignment."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from charconc.corpus import attach_labels, author_of, fold, load_author_labels


class TestAuthorNormalization:
    def test_folds_accents_and_case(self):
        assert fold("Féval") == "feval"
        assert fold("MONTÉPIN") == "montepin"

    @pytest.mark.parametrize("filename,expected", [
        ("1863_Feval_Paul_Les-Habits-noirs", "feval"),          # Gallica layout
        ("1812_Kock-Paul-de_Georgette", "kock"),                # CHAPITRES layout
        ("1839_Balzac-Honore-de_Le-Pere-Goriot", "balzac"),
    ])
    def test_extracts_surname_from_both_conventions(self, filename, expected):
        assert author_of(filename) == expected

    def test_returns_empty_on_malformed_name(self):
        assert author_of("nofields") == ""


class TestLabels:
    def test_groups_do_not_overlap(self):
        labels = load_author_labels()
        serial = {fold(a) for g in labels["feuilleton"].values() for a in g}
        control = {fold(a) for g in labels["control"].values() for a in g}
        assert not (serial & control), f"overlapping authors: {serial & control}"

    def test_excluded_authors_are_in_neither_group(self):
        labels = load_author_labels()
        assigned = {fold(a) for side in ("feuilleton", "control")
                    for g in labels[side].values() for a in g}
        excluded = {fold(a) for a in labels["excluded"]["authors"]}
        assert not (assigned & excluded)

    def test_attach_labels_marks_three_states(self):
        labels = load_author_labels()
        df = pd.DataFrame({"file_name": ["1850_Sue_X", "1857_Flaubert_Y", "1900_Unknown_Z"],
                           "year": [1850, 1857, 1900], "gini_pagerank": [.3, .5, .4],
                           "author": ["sue", "flaubert", "unknown"]})
        out = attach_labels(df, labels)
        assert out.serial.tolist()[:2] == [1.0, 0.0]
        assert pd.isna(out.serial.iloc[2])
