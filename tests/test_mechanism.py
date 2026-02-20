"""Tests for ccai_validation.mechanism (Sharded Citizens' Assembly)."""

import numpy as np
import pytest

from ccai_validation.core import is_simplex, l2_gradient, l2_utility
from ccai_validation.mechanism import citizens_assembly


class TestCABasics:
    """Structural / output-shape tests."""

    def test_output_on_simplex(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        assert is_simplex(ca.pi)

    def test_partition_disjoint(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        assert len(set(ca.A_idx) & set(ca.B_idx)) == 0

    def test_partition_complete(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        assert len(ca.A_idx) + len(ca.B_idx) == len(symmetric_5d)

    def test_axes_orthonormal(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        G = ca.axes @ ca.axes.T  # (K, K)
        assert np.allclose(G, np.eye(4), atol=1e-8)

    def test_projections_shape(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        assert ca.projections.shape == (len(ca.B_idx), 4)

    def test_medians_length(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, seed=0)
        assert len(ca.medians) == 4


class TestSharding:
    """Sharding-specific properties."""

    def test_subpanels_exist_when_sharded(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, shard=True, seed=0)
        assert ca.subpanels is not None
        assert len(ca.subpanels) == 4

    def test_subpanels_none_when_unsharded(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, shard=False, seed=0)
        assert ca.subpanels is None

    def test_subpanels_disjoint(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, shard=True, seed=0)
        all_idx = np.concatenate(ca.subpanels)
        assert len(all_idx) == len(set(all_idx)), "subpanels overlap"

    def test_subpanels_cover_panel_b(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d, K=4, shard=True, seed=0)
        all_idx = set(np.concatenate(ca.subpanels))
        assert all_idx == set(range(len(ca.B_idx)))


class TestDeterminism:
    """Same seed → same result."""

    def test_same_seed_same_output(self, symmetric_5d):
        ca1 = citizens_assembly(symmetric_5d, K=4, seed=123)
        ca2 = citizens_assembly(symmetric_5d, K=4, seed=123)
        assert np.allclose(ca1.pi, ca2.pi)
        assert np.array_equal(ca1.A_idx, ca2.A_idx)

    def test_different_seed_different_partition(self, symmetric_5d):
        ca1 = citizens_assembly(symmetric_5d, K=4, seed=0)
        ca2 = citizens_assembly(symmetric_5d, K=4, seed=1)
        assert not np.array_equal(ca1.A_idx, ca2.A_idx)


class TestPanelIndependence:
    """
    Theorem 5.2 structural guarantee: Panel B manipulation
    must not change the axes (extracted from Panel A).
    """

    def test_axes_unchanged_under_b_manipulation(self, large_ideals):
        ca_h = citizens_assembly(large_ideals, K=5, seed=7, shard=True)
        # Corrupt all Panel B agents
        ideals_m = large_ideals.copy()
        ideals_m[ca_h.B_idx] = 1.0 / 10  # uniform
        ca_m = citizens_assembly(ideals_m, K=5, seed=7, shard=True)
        assert np.allclose(ca_h.axes, ca_m.axes), "axes must depend only on Panel A"


class TestPerAxisSP:
    """
    Theorem 5.2: no single agent on a subpanel can benefit by lying
    on that axis (1D median SP).
    """

    def test_single_agent_cannot_benefit(self, large_ideals):
        ca = citizens_assembly(large_ideals, K=5, seed=42, shard=True)
        assert ca.subpanels is not None
        for k in range(5):
            pk = ca.subpanels[k]
            honest_med = np.median(ca.projections[pk, k])
            for loc_i in pk[:20]:  # sample
                true_proj = ca.projections[loc_i, k]
                # Optimal lie: far to other side
                fake = honest_med + 999 if true_proj < honest_med else honest_med - 999
                arr = ca.projections[pk, k].copy()
                arr[np.where(pk == loc_i)[0][0]] = fake
                new_med = np.median(arr)
                old_dist = abs(honest_med - true_proj)
                new_dist = abs(new_med - true_proj)
                assert new_dist >= old_dist - 1e-12, (
                    f"axis {k}, agent {loc_i}: SP violated "
                    f"(old_dist={old_dist:.6f}, new_dist={new_dist:.6f})"
                )


class TestUnanimity:
    """If all agents agree, mechanism should recover that point."""

    def test_unanimous_recovery(self):
        p = np.array([0.5, 0.3, 0.2])
        ideals = np.tile(p, (51, 1))
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)
            ca = citizens_assembly(ideals, K=2, seed=0, shard=True)
        assert np.allclose(ca.pi, p, atol=0.01)

    def test_k_defaults(self, symmetric_5d):
        ca = citizens_assembly(symmetric_5d)  # K=None → min(m-1, 5)=4
        assert ca.axes.shape[0] == 4


class TestOutputCorrectness:
    """Verify π is actually reconstructed from medians × axes."""

    def test_pi_matches_reconstruction(self, symmetric_5d):
        """π must equal to_simplex(μ_A + medians @ axes)."""
        ca = citizens_assembly(symmetric_5d, K=4, seed=0, shard=True)
        expected_raw = ca.mu_A + ca.medians @ ca.axes
        from ccai_validation.core import to_simplex
        expected = to_simplex(expected_raw)
        assert np.allclose(ca.pi, expected, atol=1e-12)

    def test_medians_are_actual_medians(self, symmetric_5d):
        """Each median_k must equal np.median of the subpanel projections."""
        ca = citizens_assembly(symmetric_5d, K=4, seed=0, shard=True)
        assert ca.subpanels is not None
        for k in range(4):
            pk = ca.subpanels[k]
            expected_med = np.median(ca.projections[pk, k])
            assert ca.medians[k] == pytest.approx(expected_med, abs=1e-14), \
                f"Axis {k}: median={ca.medians[k]} != expected={expected_med}"

    def test_nonunanimous_medians_nonzero(self, symmetric_5d):
        """For non-degenerate data, medians should not all be zero."""
        ca = citizens_assembly(symmetric_5d, K=4, seed=0, shard=True)
        assert not np.allclose(ca.medians, 0), \
            "Medians should be nonzero for non-degenerate data"

    def test_sharded_vs_unsharded_differ(self, symmetric_5d):
        """Sharded and unsharded should produce different π (different medians)."""
        ca_sh = citizens_assembly(symmetric_5d, K=4, seed=0, shard=True)
        ca_un = citizens_assembly(symmetric_5d, K=4, seed=0, shard=False)
        # Same axes and mu_A, but different medians
        assert np.allclose(ca_sh.axes, ca_un.axes)
        assert not np.allclose(ca_sh.medians, ca_un.medians), \
            "Sharding should change medians (different voter subsets per axis)"
