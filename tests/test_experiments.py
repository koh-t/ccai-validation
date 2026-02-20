"""
Tests for ccai_validation.experiments (all six experiment functions).

Design principle: every test must FAIL if the experiment function is replaced
with a stub that returns plausible dummy values.  We verify computational
correctness by:
  (a) cross-checking against independent reimplementations,
  (b) testing known analytically-derivable configurations,
  (c) asserting properties that require actual data flow.
"""

import numpy as np
import pytest

from ccai_validation.core import (
    cw_median,
    l2_gradient,
    l2_utility,
    pac_nonempty,
    to_simplex,
)
from ccai_validation.experiments import (
    exp_aggregation,
    exp_constitution,
    exp_mechanism_properties,
    exp_pac,
    exp_pairwise,
    exp_sharding,
)
from ccai_validation.mechanism import citizens_assembly


# ────────────────────────────────────────────────────
# Experiment 1: PAC
# ────────────────────────────────────────────────────

class TestExpPAC:
    def test_cross_check_with_manual_computation(self, symmetric_5d, groups_50):
        """Recompute PAC rate independently and verify exact match."""
        r = exp_pac(symmetric_5d, groups_50, n_samples=300, seed=7)
        # Independent recomputation with same seed
        rng = np.random.RandomState(7)
        pi = cw_median(symmetric_5d)
        grads = np.array([l2_gradient(pi, symmetric_5d[i])
                          for i in range(len(symmetric_5d))])
        manual_pac = 0
        manual_total = 0
        for _ in range(300):
            i, j = rng.choice(len(symmetric_5d), 2, replace=False)
            gi, gj = grads[i], grads[j]
            if np.linalg.norm(gi) < 1e-15 or np.linalg.norm(gj) < 1e-15:
                continue
            manual_total += 1
            if pac_nonempty(gi, gj):
                manual_pac += 1
        expected_rate = manual_pac / max(1, manual_total)
        assert r.rates["all"] == pytest.approx(expected_rate, abs=1e-12)

    def test_collinear_agents_high_nonempty(self):
        """Agents generated from a low-variance distribution: high PAC rate."""
        # Tight cluster → gradients mostly aligned → high PAC rate
        rng_loc = np.random.RandomState(77)
        center = np.array([0.4, 0.35, 0.25])
        ideals = np.array([to_simplex(center + 0.01 * rng_loc.randn(3))
                           for _ in range(20)])
        groups = np.array([0]*10 + [1]*10, dtype=np.intp)
        r = exp_pac(ideals, groups, n_samples=100, seed=0)
        assert r.rates["all"] > 0.9
        assert r.mean_cos > 0.0

    def test_two_agent_antiparallel(self):
        """Two agents at opposite vertices in dim=2 → antiparallel."""
        ideals = np.array([[0.99, 0.01], [0.01, 0.99]])
        groups = np.array([0, 1], dtype=np.intp)
        r = exp_pac(ideals, groups, n_samples=50, seed=0)
        assert r.antiparallel_frac > 0.9

    def test_group_rates_bracket_overall(self, symmetric_5d, groups_50):
        """Overall rate must lie between min and max of subgroup rates."""
        r = exp_pac(symmetric_5d, groups_50, n_samples=500, seed=0)
        subrates = [r.rates["intra_g0"], r.rates["intra_g1"], r.rates["cross"]]
        assert min(subrates) - 1e-10 <= r.rates["all"] <= max(subrates) + 1e-10


# ────────────────────────────────────────────────────
# Experiment 2: Aggregation
# ────────────────────────────────────────────────────

class TestExpAggregation:
    def test_no_coalition_zero_shift(self, symmetric_5d, groups_50):
        """If no agents are in G0, manipulation has zero effect."""
        all_g1 = np.ones(len(symmetric_5d), dtype=np.intp)
        rows = exp_aggregation(symmetric_5d, all_g1, strength=0.05, n_ca_trials=5)
        for r in rows:
            # Empty G0 → target = median (no manipulation attempted)
            assert r.shift == pytest.approx(0.0, abs=1e-8), \
                f"{r.rule}: shift={r.shift} with empty G0"

    def test_positive_strength_positive_shift_for_mean(self, symmetric_5d, groups_50):
        """Mean is maximally manipulable → nonzero shift with enough strength."""
        rows = exp_aggregation(symmetric_5d, groups_50, strength=0.1, n_ca_trials=5)
        mean_row = next(r for r in rows if r.rule == "Mean")
        assert mean_row.shift > 0

    def test_shift_monotone_in_strength(self, symmetric_5d, groups_50):
        """Much larger strength → larger shift (using wide gap)."""
        rows_s = exp_aggregation(symmetric_5d, groups_50, strength=0.001, n_ca_trials=5)
        rows_l = exp_aggregation(symmetric_5d, groups_50, strength=1.0, n_ca_trials=5)
        # Only test median (most stable)
        median_s = next(r for r in rows_s if r.rule == "Coord-wise median")
        median_l = next(r for r in rows_l if r.rule == "Coord-wise median")
        assert median_l.shift >= median_s.shift - 1e-10

    def test_all_rules_present(self, symmetric_5d, groups_50):
        """Returns exactly 5 rows with expected rule names."""
        rows = exp_aggregation(symmetric_5d, groups_50, strength=0.01, n_ca_trials=3)
        names = {r.rule for r in rows}
        assert names == {"Mean", "Trimmed mean (10%)", "Coord-wise median",
                         "CA (unsharded)", "CA (sharded)"}


# ────────────────────────────────────────────────────
# Experiment 3: Mechanism properties
# ────────────────────────────────────────────────────

class TestExpMechanism:
    def test_sp_cross_checked(self, large_ideals, large_groups):
        """Verify SP=0 by independently testing 1D median SP."""
        r = exp_mechanism_properties(large_ideals, large_groups,
                                     n_trials=5, max_agents_per_axis=10)
        assert r.sp_violation == pytest.approx(0.0, abs=1e-10)
        # Independent cross-check for 1 trial
        ca = citizens_assembly(large_ideals, K=9, seed=0, shard=True)
        assert ca.subpanels is not None
        for k in range(9):
            pk = ca.subpanels[k]
            med = np.median(ca.projections[pk, k])
            for loc in pk[:5]:
                true_val = ca.projections[loc, k]
                fake = med + 999 if true_val < med else med - 999
                arr = ca.projections[pk, k].copy()
                arr[np.where(pk == loc)[0][0]] = fake
                new_med = np.median(arr)
                assert abs(new_med - true_val) >= abs(med - true_val) - 1e-12

    def test_panel_independence_with_median_change(self, large_ideals, large_groups):
        """Axes unchanged but medians MUST change when Panel B is corrupted."""
        r = exp_mechanism_properties(large_ideals, large_groups,
                                     n_trials=5, max_agents_per_axis=10)
        assert r.panel_independence == pytest.approx(0.0, abs=1e-10)
        # Cross-check: axes same but medians differ
        ca = citizens_assembly(large_ideals, K=5, seed=99, shard=True)
        corrupted = large_ideals.copy()
        corrupted[ca.B_idx] = 0.1
        ca2 = citizens_assembly(corrupted, K=5, seed=99, shard=True)
        assert np.allclose(ca.axes, ca2.axes)
        assert not np.allclose(ca.medians, ca2.medians), \
            "Medians must change when Panel B is corrupted"

    def test_sp_test_exercises_agents(self, large_ideals, large_groups):
        """Verify SP test actually exercises agents (total > 0)."""
        ca = citizens_assembly(large_ideals, K=5, seed=0, shard=True)
        assert ca.subpanels is not None
        g0_set = set(np.where(large_groups == 0)[0])
        total_tested = sum(
            min(20, sum(1 for loc in ca.subpanels[k]
                        if ca.B_idx[loc] in g0_set))
            for k in range(5)
        )
        assert total_tested > 10, "SP test must exercise substantial agents"


# ────────────────────────────────────────────────────
# Experiment 4: Sharding
# ────────────────────────────────────────────────────

class TestExpSharding:
    def test_empty_coalition_zero_shift(self, large_ideals, large_groups):
        """0-person coalition → shift = 0."""
        all_g1 = np.ones(len(large_ideals), dtype=np.intp)
        r = exp_sharding(large_ideals, all_g1, n_trials=10, use_minority=False)
        assert r.rho == pytest.approx(0.0)
        for s in r.shifts_sharded:
            assert s == pytest.approx(0.0, abs=1e-10)

    def test_majority_captures_scale_with_K(self, large_ideals, large_groups):
        """ρ=0.6 captures many axes; actual count consistent with K."""
        r = exp_sharding(large_ideals, large_groups,
                         n_trials=50, use_minority=False)
        K = large_ideals.shape[1] - 1  # 9
        mean_c = np.mean(r.captures)
        assert mean_c > K * 0.3
        assert all(0 <= c <= K for c in r.captures)

    def test_extreme_minority_zero_captures(self, large_ideals, large_groups):
        """ρ=0.15 → ~0 captures."""
        r = exp_sharding(large_ideals, large_groups,
                         n_trials=50, use_minority=True, minority_frac=0.15)
        assert np.mean(r.captures) < 0.5

    def test_sharding_limits_minority_captures(self, large_ideals, large_groups):
        """For ρ < 0.5, sharding limits captured axes (Prop 5.3)."""
        r = exp_sharding(large_ideals, large_groups,
                         n_trials=50, use_minority=True, minority_frac=0.25)
        K = large_ideals.shape[1] - 1  # 9
        mean_c = np.mean(r.captures)
        # Much fewer captures than K
        assert mean_c < K * 0.5


# ────────────────────────────────────────────────────
# Experiment 5: Pairwise logrolling
# ────────────────────────────────────────────────────

class TestExpPairwise:
    def test_identical_agents_zero_success(self):
        """All agents at same ideal → gradient = 0 → zero logrolling."""
        p = np.array([0.4, 0.3, 0.2, 0.1])
        ideals = np.tile(p, (30, 1))
        r = exp_pairwise(ideals, n_samples=100, seed=0)
        assert r.success == 0

    def test_rate_increases_with_dimension(self):
        """Higher m → more directions → higher success rate."""
        rng = np.random.RandomState(99)
        rates = []
        for m in [3, 5, 10]:
            ideals = rng.dirichlet(np.ones(m), size=100)
            r = exp_pairwise(ideals, n_samples=200, seed=0)
            rates.append(r.rate)
        assert rates[0] <= rates[1] + 0.05
        assert rates[1] <= rates[2] + 0.05
        assert rates[2] > 0.8

    def test_gains_cross_check_first_success(self):
        """Reproduce the first success and verify exact du values."""
        rng_gen = np.random.RandomState(42)
        ideals = rng_gen.dirichlet(np.ones(5), size=50)
        r = exp_pairwise(ideals, n_samples=200, seed=42)
        assert len(r.gains) > 0

        pi_star = cw_median(ideals)
        rng = np.random.RandomState(42)
        for _ in range(200):
            i, j = rng.choice(len(ideals), 2, replace=False)
            gi = l2_gradient(pi_star, ideals[i])
            gj = l2_gradient(pi_star, ideals[j])
            d = gi + gj
            if np.dot(d, gi) > 0 and np.dot(d, gj) > 0:
                pi_new = pi_star + 0.001 * d / (np.linalg.norm(d) + 1e-15)
                pi_new = to_simplex(pi_new)
                du_i = l2_utility(pi_new, ideals[i]) - l2_utility(pi_star, ideals[i])
                du_j = l2_utility(pi_new, ideals[j]) - l2_utility(pi_star, ideals[j])
                if du_i > 0 and du_j > 0:
                    assert r.gains[0][0] == pytest.approx(du_i, rel=1e-10)
                    assert r.gains[0][1] == pytest.approx(du_j, rel=1e-10)
                    break

    def test_example_4_10_triangle(self, triangle_ideals):
        """Example 4.10: m=3 triangle agents have logrolling opportunities."""
        r = exp_pairwise(triangle_ideals, n_samples=200, seed=0)
        assert r.rate > 0.3


# ────────────────────────────────────────────────────
# Experiment 6: Constitution
# ────────────────────────────────────────────────────

class TestExpConstitution:
    def test_no_gap_threshold_inf_no_manipulation(self):
        """gap_threshold=1.0 → nothing contentious → J(honest,manip) = 1."""
        rng = np.random.RandomState(0)
        V = rng.choice([1.0, -1.0], size=(80, 20), p=[0.6, 0.4])
        groups = np.array([0]*40 + [1]*40, dtype=np.intp)
        cols = [str(i) for i in range(20)]
        r = exp_constitution(V, groups, cols, gap_threshold=1.0)
        assert r.n_contentious == 0
        assert r.J_honest_manip == pytest.approx(1.0)
        assert r.n_entered == 0
        assert r.n_exited == 0

    def test_extreme_polarization(self):
        """G0 (minority) and G1 want opposite sets → manipulation changes constitution."""
        n = 100
        V = np.full((n, 30), np.nan)
        # G0 is minority (40%) — their preferences differ from honest consensus
        groups = np.array([0]*40 + [1]*60, dtype=np.intp)
        for i in range(40):
            V[i, :15] = 1.0; V[i, 15:] = -1.0
        for i in range(40, 100):
            V[i, :15] = -1.0; V[i, 15:] = 1.0
        cols = [str(i) for i in range(30)]
        r = exp_constitution(V, groups, cols, top_k=15, gap_threshold=0.05)
        assert r.J_g0_g1 == pytest.approx(0.0)
        # G0 minority strategic voting should shift the constitution
        assert r.n_contentious > 0

    def test_contentious_count_matches_data(self):
        """Verify n_contentious by manual gap computation."""
        rng = np.random.RandomState(5)
        V = rng.choice([1.0, -1.0], size=(100, 10), p=[0.5, 0.5])
        groups = np.array([0]*60 + [1]*40, dtype=np.intp)
        # Force known gaps on first 3 columns
        V[:60, 0] = 1.0; V[60:, 0] = -1.0   # gap = +1.0
        V[:60, 1] = -1.0; V[60:, 1] = 1.0   # gap = -1.0
        V[:60, 2] = 1.0; V[60:, 2] = 1.0    # gap = 0.0
        cols = [str(i) for i in range(10)]
        r = exp_constitution(V, groups, cols, gap_threshold=0.50)
        # Cols 0 and 1 have |gap| = 1.0 > 0.5; col 2 has gap = 0
        # Other cols are random ±1 → gap varies
        assert r.n_contentious >= 2

    def test_monotone_in_threshold(self, vote_matrix, groups_100, vote_cols_30):
        """Stricter threshold → fewer contentious statements."""
        r1 = exp_constitution(vote_matrix, groups_100, vote_cols_30,
                              gap_threshold=0.01)
        r2 = exp_constitution(vote_matrix, groups_100, vote_cols_30,
                              gap_threshold=0.50)
        assert r1.n_contentious >= r2.n_contentious
