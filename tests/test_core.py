"""Tests for ccai_validation.core."""

import numpy as np
import pytest

from ccai_validation.core import (
    coalition_target,
    cw_mean,
    cw_median,
    cw_trimmed_mean,
    embed_simplex,
    gradient_cos,
    impute_votes,
    is_simplex,
    jaccard,
    l2_gradient,
    l2_sq,
    l2_utility,
    pac_nonempty,
    to_simplex,
    utility_gain,
)


# ────────────────────────────────────────────────────
# Simplex operations
# ────────────────────────────────────────────────────

class TestSimplex:
    def test_to_simplex_sums_to_one(self):
        x = np.array([2.0, 3.0, 5.0])
        s = to_simplex(x)
        assert abs(s.sum() - 1.0) < 1e-12

    def test_to_simplex_positive(self):
        x = np.array([-0.1, 0.5, 0.3])
        s = to_simplex(x)
        assert np.all(s > 0)

    def test_is_simplex_valid(self):
        assert is_simplex(np.array([0.3, 0.3, 0.4]))

    def test_is_simplex_invalid_sum(self):
        assert not is_simplex(np.array([0.5, 0.5, 0.5]))

    def test_is_simplex_negative(self):
        assert not is_simplex(np.array([-0.1, 0.6, 0.5]))


# ────────────────────────────────────────────────────
# Embedding
# ────────────────────────────────────────────────────

class TestEmbedding:
    def test_output_on_simplex(self, vote_matrix):
        V_filled = np.nan_to_num(vote_matrix, nan=0.0)
        ideals, pca = embed_simplex(V_filled, m=5)
        assert ideals.shape == (100, 5)
        for i in range(100):
            assert is_simplex(ideals[i]), f"agent {i} not on simplex"

    def test_pca_components(self, vote_matrix):
        V_filled = np.nan_to_num(vote_matrix, nan=0.0)
        _, pca = embed_simplex(V_filled, m=3)
        assert pca.n_components == 3

    def test_impute_votes_no_nan(self, vote_matrix):
        filled = impute_votes(vote_matrix, fill=0.0)
        assert not np.any(np.isnan(filled))

    def test_impute_preserves_shape(self, vote_matrix):
        filled = impute_votes(vote_matrix)
        assert filled.shape == vote_matrix.shape


# ────────────────────────────────────────────────────
# L2 utility & gradient
# ────────────────────────────────────────────────────

class TestL2:
    def test_l2_sq_zero(self):
        a = np.array([0.3, 0.3, 0.4])
        assert l2_sq(a, a) == pytest.approx(0.0)

    def test_l2_sq_symmetric(self):
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.1, 0.4, 0.5])
        assert l2_sq(a, b) == pytest.approx(l2_sq(b, a))

    def test_utility_maximised_at_ideal(self):
        p = np.array([0.4, 0.4, 0.2])
        # utility at own ideal is 0 (maximum)
        assert l2_utility(p, p) == pytest.approx(0.0)

    def test_utility_negative_elsewhere(self):
        p = np.array([0.4, 0.4, 0.2])
        pi = np.array([0.1, 0.1, 0.8])
        assert l2_utility(pi, p) < 0.0

    def test_gradient_in_tangent_space(self):
        """∇u must lie in T∆: components sum to 0."""
        pi = np.array([1 / 3, 1 / 3, 1 / 3])
        p = np.array([0.6, 0.2, 0.2])
        g = l2_gradient(pi, p)
        assert abs(g.sum()) < 1e-12

    def test_gradient_exact_value(self):
        """∇u = 2(p_i - π) must hold exactly."""
        pi = np.array([1 / 3, 1 / 3, 1 / 3])
        p = np.array([0.8, 0.1, 0.1])
        g = l2_gradient(pi, p)
        expected = 2.0 * (p - pi)
        assert np.allclose(g, expected, atol=1e-14)

    def test_gradient_zero_at_ideal(self):
        p = np.array([0.4, 0.4, 0.2])
        g = l2_gradient(p, p)
        assert np.allclose(g, 0)


# ────────────────────────────────────────────────────
# Aggregation rules
# ────────────────────────────────────────────────────

class TestAggregation:
    def test_median_on_simplex(self, triangle_ideals):
        m = cw_median(triangle_ideals)
        assert is_simplex(m)

    def test_mean_on_simplex(self, triangle_ideals):
        m = cw_mean(triangle_ideals)
        assert is_simplex(m)

    def test_trimmed_mean_on_simplex(self, triangle_ideals):
        m = cw_trimmed_mean(triangle_ideals, trim=0.1)
        assert is_simplex(m)

    def test_unanimous_median(self):
        """If all agents agree, median = ideal point."""
        p = np.array([0.5, 0.3, 0.2])
        ideals = np.tile(p, (11, 1))
        m = cw_median(ideals)
        assert np.allclose(m, p, atol=1e-10)

    def test_unanimous_mean(self):
        p = np.array([0.5, 0.3, 0.2])
        ideals = np.tile(p, (10, 1))
        m = cw_mean(ideals)
        assert np.allclose(m, p, atol=1e-10)

    def test_mean_equals_centroid(self, triangle_ideals):
        """For Example 4.10, mean is the centroid (1/3,1/3,1/3)."""
        m = cw_mean(triangle_ideals)
        assert np.allclose(m, [1 / 3, 1 / 3, 1 / 3], atol=0.02)


# ────────────────────────────────────────────────────
# PAC
# ────────────────────────────────────────────────────

class TestPAC:
    def test_parallel_nonempty(self):
        """Parallel gradients → PAC nonempty."""
        a = np.array([1.0, -1.0, 0.0])
        assert pac_nonempty(a, a) is True

    def test_antiparallel_empty(self):
        """Antiparallel gradients → PAC empty."""
        a = np.array([1.0, -1.0, 0.0])
        assert pac_nonempty(a, -a) is False

    def test_orthogonal_nonempty(self):
        """Orthogonal gradients in dim≥2 → PAC nonempty."""
        a = np.array([1.0, -1.0, 0.0])
        b = np.array([0.0, 1.0, -1.0])
        assert pac_nonempty(a, b) is True

    def test_zero_gradient_empty(self):
        a = np.array([1.0, -1.0, 0.0])
        z = np.zeros(3)
        assert pac_nonempty(a, z) is False

    def test_1d_straddling_empty(self):
        """In dim=1, agents on opposite sides → antiparallel → empty."""
        a = np.array([1.0])
        b = np.array([-1.0])
        assert pac_nonempty(a, b) is False

    def test_example_4_10(self, triangle_ideals):
        """Example 4.10: agents near distinct vertices, π*=centroid."""
        pi_star = np.array([1 / 3, 1 / 3, 1 / 3])
        g1 = l2_gradient(pi_star, triangle_ideals[0])
        g2 = l2_gradient(pi_star, triangle_ideals[1])
        # cos = -1/(m-1) = -0.5 for m=3, not antiparallel
        cos_val = gradient_cos(g1, g2)
        assert cos_val == pytest.approx(-0.5, abs=0.01)
        assert pac_nonempty(g1, g2) is True


# ────────────────────────────────────────────────────
# Coalition helpers
# ────────────────────────────────────────────────────

class TestCoalition:
    def test_target_on_simplex(self, symmetric_5d, groups_50):
        g0_idx = np.where(groups_50 == 0)[0]
        pi = cw_median(symmetric_5d)
        t = coalition_target(symmetric_5d, g0_idx, pi, 0.01)
        assert is_simplex(t)

    def test_target_differs_from_median(self, symmetric_5d, groups_50):
        g0_idx = np.where(groups_50 == 0)[0]
        pi = cw_median(symmetric_5d)
        t = coalition_target(symmetric_5d, g0_idx, pi, 0.01)
        assert not np.allclose(t, pi)

    def test_empty_coalition_returns_pi(self, symmetric_5d):
        """Empty coalition → target = π* (no NaN)."""
        pi = cw_median(symmetric_5d)
        t = coalition_target(symmetric_5d, np.array([], dtype=np.intp), pi, 0.01)
        assert np.allclose(t, pi)
        assert not np.any(np.isnan(t))

    def test_utility_gain_unanimous(self):
        """If π moves towards all agents, all gain."""
        ideals = np.array([[0.6, 0.2, 0.2], [0.7, 0.15, 0.15]])
        pi_old = np.array([1 / 3, 1 / 3, 1 / 3])
        pi_new = np.array([0.5, 0.25, 0.25])
        du, frac = utility_gain(pi_new, pi_old, ideals, np.array([0, 1]))
        assert du > 0
        assert frac == pytest.approx(1.0)


# ────────────────────────────────────────────────────
# Jaccard
# ────────────────────────────────────────────────────

class TestJaccard:
    def test_identical(self):
        assert jaccard({1, 2, 3}, {1, 2, 3}) == pytest.approx(1.0)

    def test_disjoint(self):
        assert jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)

    def test_partial(self):
        assert jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)

    def test_empty(self):
        assert jaccard(set(), set()) == pytest.approx(0.0)
