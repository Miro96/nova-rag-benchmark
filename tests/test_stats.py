"""Tests for rag_bench.stats — pure-math statistical primitives.

Covers wilson_ci, normal_ci, pearson_correlation_matrix with analytically
known reference values.
"""
from __future__ import annotations

from math import isclose

import numpy as np
import pytest

from rag_bench.stats import wilson_ci, normal_ci, pearson_correlation_matrix


# -----------------------------------------------------------------------
# wilson_ci
# -----------------------------------------------------------------------

class TestWilsonCI:
    """Wilson 95 % CI for proportions."""

    def test_wilson_5_of_35(self):
        """5/35 hits → Wilson 95% CI ≈ [0.0626, 0.2938]."""
        lo, hi = wilson_ci(5, 35)
        assert abs(lo - 0.0626) < 0.0001
        assert abs(hi - 0.2938) < 0.0001

    def test_wilson_35_of_35(self):
        """All hits → CI is [0.9011, 1.0] (or very close)."""
        lo, hi = wilson_ci(35, 35)
        assert abs(lo - 0.9011) < 0.001
        assert abs(hi - 1.0) < 1e-6

    def test_wilson_0_of_35(self):
        """No hits → CI is [0.0, 0.0989]."""
        lo, hi = wilson_ci(0, 35)
        assert abs(lo - 0.0) < 1e-12
        assert abs(hi - 0.0989) < 0.001

    def test_wilson_symmetry(self):
        """Wilson(k, n) and Wilson(n-k, n) are symmetric."""
        ci_pos = wilson_ci(5, 35)
        ci_neg = wilson_ci(30, 35)
        # Symmetry: ci_low(5,35) ≈ 1 - ci_high(30,35) and vice versa
        assert abs(ci_pos[0] - (1.0 - ci_neg[1])) < 1e-9
        assert abs(ci_pos[1] - (1.0 - ci_neg[0])) < 1e-9

    def test_wilson_1_of_1(self):
        """Single success → CI brackets around 1.0."""
        lo, hi = wilson_ci(1, 1)
        assert 0.0 <= lo < 1.0
        assert hi == 1.0
        assert lo <= hi

    def test_wilson_custom_alpha(self):
        """alpha=0.10 yields wider interval than alpha=0.05."""
        ci_95 = wilson_ci(10, 20, alpha=0.05)
        ci_90 = wilson_ci(10, 20, alpha=0.10)
        width_95 = ci_95[1] - ci_95[0]
        width_90 = ci_90[1] - ci_90[0]
        assert width_95 > width_90

    def test_wilson_raises_n_zero(self):
        """n <= 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(5, 0)

    def test_wilson_raises_n_negative(self):
        """n < 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(5, -3)

    def test_wilson_raises_k_negative(self):
        """k < 0 → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)

    def test_wilson_raises_k_gt_n(self):
        """k > n → ValueError."""
        with pytest.raises(ValueError):
            wilson_ci(50, 10)

    def test_accepts_tuple(self):
        """Python tuples work for k and n."""
        ci = wilson_ci(5, 35)
        assert isinstance(ci, tuple)
        assert len(ci) == 2


# -----------------------------------------------------------------------
# normal_ci
# -----------------------------------------------------------------------

class TestNormalCI:
    """Normal/t CI for means."""

    def test_normal_known_bounds(self):
        """mean=100, std=15, n=30 → CI ≈ (94.45, 105.55) (t, df=29)."""
        lo, hi = normal_ci(100, 15, 30)
        # Reference: t.ppf(0.975, 29) ≈ 2.045; margin ≈ 2.045*15/sqrt(30) ≈ 5.55
        assert abs(lo - 94.45) < 0.1
        assert abs(hi - 105.55) < 0.1

    def test_normal_symmetric(self):
        """CI bounds are symmetric around the mean."""
        lo, hi = normal_ci(50, 10, 50)
        assert abs((lo + hi) / 2 - 50.0) < 1e-9

    def test_normal_std_zero(self):
        """std=0 → CI collapses to mean."""
        lo, hi = normal_ci(42.0, 0.0, 20)
        assert lo == 42.0
        assert hi == 42.0

    def test_normal_n_1(self):
        """n=1 → uses normal (not t), symmetric."""
        lo, hi = normal_ci(10.0, 5.0, 1)
        assert lo < 10.0 < hi
        # t with df=0 falls back to norm

    def test_normal_custom_alpha(self):
        """alpha=0.01 yields wider interval than alpha=0.05."""
        ci_95 = normal_ci(100, 15, 30, alpha=0.05)
        ci_99 = normal_ci(100, 15, 30, alpha=0.01)
        assert (ci_99[1] - ci_99[0]) > (ci_95[1] - ci_95[0])

    def test_raises_n_zero(self):
        """n <= 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, 5, 0)

    def test_raises_n_negative(self):
        """n < 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, 5, -5)

    def test_raises_std_negative(self):
        """std < 0 → ValueError."""
        with pytest.raises(ValueError):
            normal_ci(10, -3, 20)


# -----------------------------------------------------------------------
# pearson_correlation_matrix
# -----------------------------------------------------------------------

class TestPearsonCorrelationMatrix:
    """Pearson correlation matrix via numpy.corrcoef."""

    def test_perfect_positive(self):
        """Perfectly correlated (y = 2x) → correlation 1.0."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1] - 1.0) < 1e-9

    def test_perfect_negative(self):
        """Perfectly anti-correlated (y = -x) → correlation -1.0."""
        x = [1, 2, 3, 4, 5]
        y = [-1, -2, -3, -4, -5]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1] + 1.0) < 1e-9

    def test_uncorrelated(self):
        """Orthogonal vectors → near-zero correlation."""
        x = [1, -1, 0, 0]
        y = [0, 0, 1, -1]
        m = pearson_correlation_matrix(x, y)
        assert abs(m[0, 1]) < 1e-12

    def test_diagonal_is_one(self):
        """Diagonal entries are exactly 1.0."""
        x = [1.0, 2.0, 3.0]
        y = [3.0, 2.0, 1.0]
        z = [10, 20, 30]
        m = pearson_correlation_matrix(x, y, z)
        for i in range(3):
            assert m[i, i] == 1.0

    def test_symmetry(self):
        """Matrix is symmetric within 1e-9."""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 1, 4, 3, 6, 5, 8, 7, 9, 10]
        z = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        m = pearson_correlation_matrix(x, y, z)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(m[i, j] - m[j, i]) < 1e-9

    def test_shape(self):
        """Matrix shape is (k, k) for k columns."""
        cols = [[1, 2, 3]] * 5
        m = pearson_correlation_matrix(*cols)
        assert m.shape == (5, 5)

    def test_accepts_tuples(self):
        """Python tuples work as input."""
        x = (1, 2, 3)
        y = (3, 2, 1)
        m = pearson_correlation_matrix(x, y)
        assert m.shape == (2, 2)

    def test_raises_unequal_length(self):
        """Different lengths → ValueError."""
        with pytest.raises(ValueError):
            pearson_correlation_matrix([1, 2], [1, 2, 3])

    def test_raises_empty(self):
        """Empty arrays → ValueError."""
        with pytest.raises(ValueError):
            pearson_correlation_matrix([], [])
