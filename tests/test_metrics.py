"""Unit tests for the concentration indices."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from charconc.metrics import concentration_ratio, gini, hhi, normalized_entropy


class TestGini:
    def test_perfect_equality_is_zero(self):
        assert gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)

    def test_maximal_inequality_approaches_one(self):
        assert gini([0, 0, 0, 100]) == pytest.approx(0.75, abs=1e-9)

    def test_scale_invariance(self):
        assert gini([1, 2, 3]) == pytest.approx(gini([10, 20, 30]))

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError):
            gini([-1, 2])

    def test_all_zero_returns_zero(self):
        assert gini([0, 0, 0]) == 0.0

    def test_one_versus_two_protagonists(self):
        """The paper's interpretive anchor: with a cast of twenty, moving from
        two co-equal protagonists to one shifts the Gini by about 0.05."""
        def cast(n_leads, lead_share=0.5, total=20):
            leads = [lead_share / n_leads] * n_leads
            minors = [(1 - lead_share) / (total - n_leads)] * (total - n_leads)
            return leads + minors
        assert gini(cast(1)) - gini(cast(2)) == pytest.approx(0.05, abs=0.005)


class TestConcentrationRatio:
    def test_top_three_share(self):
        assert concentration_ratio([4, 3, 2, 1], k=3) == pytest.approx(0.9)

    def test_k_larger_than_cast(self):
        assert concentration_ratio([1, 1], k=5) == pytest.approx(1.0)


class TestOtherIndices:
    def test_hhi_bounds(self):
        assert hhi([1, 1, 1, 1]) == pytest.approx(0.25)
        assert hhi([1, 0, 0, 0]) == pytest.approx(1.0)

    def test_entropy_is_one_when_uniform(self):
        assert normalized_entropy([1, 1, 1, 1]) == pytest.approx(1.0)

    def test_entropy_falls_as_concentration_rises(self):
        assert normalized_entropy([10, 1, 1, 1]) < normalized_entropy([3, 3, 3, 2])
