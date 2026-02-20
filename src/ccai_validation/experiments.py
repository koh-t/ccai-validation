"""
Six experiments validating the paper's theorems on CCAI data.

Each ``exp_*`` function is self-contained: it takes preprocessed data
and returns a plain dict (or dataclass) of results, with no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import (
    EPS, Mat, Vec,
    cw_mean, cw_median, cw_trimmed_mean,
    coalition_target, embed_simplex,
    gradient_cos, jaccard,
    l2_gradient, l2_sq, l2_utility,
    pac_nonempty, to_simplex, utility_gain,
)
from .mechanism import citizens_assembly

# Default hyper-parameters (overridable via function args)
SEED = 42


# ================================================================
# Experiment 1: PAC nonemptiness (Lemma 4.3)
# ================================================================

@dataclass
class PACResult:
    rates: dict[str, float]
    antiparallel_frac: float
    mean_cos: float


def exp_pac(
    ideals: Mat,
    groups: NDArray[np.intp],
    n_samples: int = 2000,
    seed: int = SEED,
) -> PACResult:
    """Sample agent pairs and test PAC nonemptiness."""
    rng = np.random.RandomState(seed)
    pi_star = cw_median(ideals)
    grads = np.array([l2_gradient(pi_star, ideals[i]) for i in range(len(ideals))])

    g0 = groups == 0
    g1 = groups == 1
    results = {k: 0 for k in ["all", "intra_g0", "intra_g1", "cross"]}
    counts  = {k: 0 for k in ["all", "intra_g0", "intra_g1", "cross"]}
    antipar = 0
    cos_vals: list[float] = []

    for _ in range(n_samples):
        i, j = rng.choice(len(ideals), 2, replace=False)
        gi, gj = grads[i], grads[j]
        if np.linalg.norm(gi) < EPS or np.linalg.norm(gj) < EPS:
            continue

        cos_ij = gradient_cos(gi, gj)
        cos_vals.append(cos_ij)
        pac = pac_nonempty(gi, gj)

        counts["all"] += 1
        if pac:
            results["all"] += 1
        if cos_ij < -0.9:
            antipar += 1

        if g0[i] and g0[j]:
            counts["intra_g0"] += 1
            if pac:
                results["intra_g0"] += 1
        elif g1[i] and g1[j]:
            counts["intra_g1"] += 1
            if pac:
                results["intra_g1"] += 1
        else:
            counts["cross"] += 1
            if pac:
                results["cross"] += 1

    rates = {k: results[k] / max(1, counts[k]) for k in results}
    return PACResult(
        rates=rates,
        antiparallel_frac=antipar / max(1, counts["all"]),
        mean_cos=float(np.mean(cos_vals)) if cos_vals else 0.0,
    )


# ================================================================
# Experiment 2: Aggregation rule comparison (Theorem 4.8)
# ================================================================

@dataclass
class AggRow:
    rule: str
    shift: float
    shift_std: float | None
    g0_du: float
    g0_frac: float


def exp_aggregation(
    ideals: Mat,
    groups: NDArray[np.intp],
    strength: float = 0.01,
    n_ca_trials: int = 50,
) -> list[AggRow]:
    m = ideals.shape[1]
    g0_idx = np.where(groups == 0)[0]
    pi_honest = cw_median(ideals)
    target = coalition_target(ideals, g0_idx, pi_honest, strength)
    ideals_m = ideals.copy()
    ideals_m[g0_idx] = target

    rows: list[AggRow] = []
    for name, fn in [
        ("Mean", cw_mean),
        ("Trimmed mean (10%)", lambda x: cw_trimmed_mean(x, 0.1)),
        ("Coord-wise median", cw_median),
    ]:
        pi_h, pi_m = fn(ideals), fn(ideals_m)
        du, frac = utility_gain(pi_m, pi_h, ideals, g0_idx)
        rows.append(AggRow(name, l2_sq(pi_m, pi_h), None, du, frac))

    for shard, sname in [(False, "CA (unsharded)"), (True, "CA (sharded)")]:
        shifts, dus, fracs = [], [], []
        for t in range(n_ca_trials):
            ca_h = citizens_assembly(ideals, seed=t, K=m - 1, shard=shard)
            ideals_ca = ideals.copy()
            B_g0 = np.intersect1d(g0_idx, ca_h.B_idx)
            ideals_ca[B_g0] = target
            ca_m = citizens_assembly(ideals_ca, seed=t, K=m - 1, shard=shard)
            shifts.append(l2_sq(ca_m.pi, ca_h.pi))
            du, frac = utility_gain(ca_m.pi, ca_h.pi, ideals, g0_idx)
            dus.append(du)
            fracs.append(frac)
        rows.append(AggRow(sname, float(np.mean(shifts)),
                           float(np.std(shifts)),
                           float(np.mean(dus)), float(np.mean(fracs))))
    return rows


# ================================================================
# Experiment 3: Mechanism properties (Theorem 5.2)
# ================================================================

@dataclass
class MechResult:
    sp_violation: float
    panel_independence: float


def exp_mechanism_properties(
    ideals: Mat,
    groups: NDArray[np.intp],
    n_trials: int = 30,
    max_agents_per_axis: int = 30,
) -> MechResult:
    m = ideals.shape[1]
    K = m - 1
    g0_idx = np.where(groups == 0)[0]
    g0_set = set(g0_idx)

    # --- Per-axis SP ---
    viol_rates: list[float] = []
    for trial in range(n_trials):
        ca = citizens_assembly(ideals, seed=trial, K=K, shard=True)
        assert ca.subpanels is not None
        viol = total = 0
        for k in range(K):
            pk = ca.subpanels[k]
            honest_med = np.median(ca.projections[pk, k])
            g0_in_pk = [loc for loc in pk if ca.B_idx[loc] in g0_set]
            for loc_i in g0_in_pk[:max_agents_per_axis]:
                true_proj = ca.projections[loc_i, k]
                fake = (honest_med + 100.0 if true_proj < honest_med
                        else honest_med - 100.0)
                arr = ca.projections[pk, k].copy()
                arr[np.where(pk == loc_i)[0][0]] = fake
                new_med = np.median(arr)
                total += 1
                if abs(new_med - true_proj) < abs(honest_med - true_proj) - 1e-12:
                    viol += 1
        viol_rates.append(viol / max(1, total))

    # --- Panel independence ---
    diffs: list[float] = []
    for trial in range(n_trials):
        ca_h = citizens_assembly(ideals, seed=trial, K=K, shard=True)
        ideals_m = ideals.copy()
        B_g0 = np.intersect1d(g0_idx, ca_h.B_idx)
        ideals_m[B_g0] = np.ones(m) / m
        ca_m = citizens_assembly(ideals_m, seed=trial, K=K, shard=True)
        diffs.append(float(np.linalg.norm(ca_h.axes - ca_m.axes)))

    return MechResult(
        sp_violation=float(np.mean(viol_rates)),
        panel_independence=float(np.mean(diffs)),
    )


# ================================================================
# Experiment 4: Sharding effect (Proposition 5.3)
# ================================================================

@dataclass
class ShardResult:
    shifts_sharded: list[float]
    shifts_unsharded: list[float]
    gains_sharded: list[float]
    gains_unsharded: list[float]
    captures: list[int]
    rho: float


def exp_sharding(
    ideals: Mat,
    groups: NDArray[np.intp],
    n_trials: int = 500,
    use_minority: bool = False,
    minority_frac: float = 0.30,
    seed: int = SEED,
) -> ShardResult:
    m = ideals.shape[1]
    K = m - 1
    g0_idx = np.where(groups == 0)[0]
    rng = np.random.RandomState(seed + 1000)

    if use_minority:
        coal_size = int(len(ideals) * minority_frac)
        coal_idx = rng.choice(len(ideals), coal_size, replace=False)
        rho = minority_frac
    else:
        coal_idx = g0_idx
        rho = len(g0_idx) / len(ideals)

    coal_set = set(coal_idx)
    pi_honest = cw_median(ideals)
    target = coalition_target(ideals, coal_idx, pi_honest, 0.01)

    out: dict[str, list] = {
        "sh_shifts": [], "un_shifts": [],
        "sh_gains": [], "un_gains": [],
        "captures": [],
    }

    for trial in range(n_trials):
        for sf, s_key, g_key in [
            (True, "sh_shifts", "sh_gains"),
            (False, "un_shifts", "un_gains"),
        ]:
            ca_h = citizens_assembly(ideals, seed=trial, K=K, shard=sf)
            ideals_ca = ideals.copy()
            B_coal = np.intersect1d(coal_idx, ca_h.B_idx)
            ideals_ca[B_coal] = target
            ca_m = citizens_assembly(ideals_ca, seed=trial, K=K, shard=sf)

            out[s_key].append(l2_sq(ca_m.pi, ca_h.pi))
            _, frac = utility_gain(ca_m.pi, ca_h.pi, ideals, coal_idx)
            out[g_key].append(frac)

            if sf and ca_h.subpanels is not None:
                nc = sum(
                    1 for k in range(K)
                    if sum(1 for loc in ca_h.subpanels[k]
                           if ca_h.B_idx[loc] in coal_set) > len(ca_h.subpanels[k]) / 2
                )
                out["captures"].append(nc)

    return ShardResult(
        shifts_sharded=out["sh_shifts"],
        shifts_unsharded=out["un_shifts"],
        gains_sharded=out["sh_gains"],
        gains_unsharded=out["un_gains"],
        captures=out["captures"],
        rho=rho,
    )


# ================================================================
# Experiment 5: Pairwise logrolling (Example 4.10)
# ================================================================

@dataclass
class PairwiseResult:
    success: int
    total: int
    gains: list[tuple[float, float]]

    @property
    def rate(self) -> float:
        return self.success / max(1, self.total)


def exp_pairwise(
    ideals: Mat,
    n_samples: int = 500,
    seed: int = SEED,
) -> PairwiseResult:
    rng = np.random.RandomState(seed)
    pi_star = cw_median(ideals)

    success = 0
    gains: list[tuple[float, float]] = []
    for _ in range(n_samples):
        i, j = rng.choice(len(ideals), 2, replace=False)
        gi = l2_gradient(pi_star, ideals[i])
        gj = l2_gradient(pi_star, ideals[j])
        d = gi + gj
        if np.dot(d, gi) > 0 and np.dot(d, gj) > 0:
            delta = 0.001
            pi_new = pi_star + delta * d / (np.linalg.norm(d) + EPS)
            pi_new = to_simplex(pi_new)
            du_i = l2_utility(pi_new, ideals[i]) - l2_utility(pi_star, ideals[i])
            du_j = l2_utility(pi_new, ideals[j]) - l2_utility(pi_star, ideals[j])
            if du_i > 0 and du_j > 0:
                success += 1
                gains.append((du_i, du_j))
    return PairwiseResult(success, n_samples, gains)


# ================================================================
# Experiment 6: Constitution-level manipulation (§6.7)
# ================================================================

@dataclass
class ConstitutionResult:
    n_contentious: int
    J_honest_g0: float
    J_honest_g1: float
    J_g0_g1: float
    J_honest_manip: float
    n_entered: int
    n_exited: int


def exp_constitution(
    V_raw: Mat,
    groups: NDArray[np.intp],
    vote_cols: list[str],
    top_k: int = 15,
    gap_threshold: float = 0.05,
) -> ConstitutionResult:
    """NaN-aware constitution analysis."""
    g0_mask = groups == 0
    g1_mask = groups == 1
    g0_idx = np.where(g0_mask)[0]

    # Per-statement stats
    stats: list[dict] = []
    for idx, col in enumerate(vote_cols):
        v = V_raw[:, idx]
        valid = ~np.isnan(v)
        n_v = int(valid.sum())
        if n_v < 10:
            continue
        agree = float((v[valid] == 1).sum() / n_v)
        g0_v = v[valid & g0_mask]
        g1_v = v[valid & g1_mask]
        g0a = float((g0_v == 1).sum() / max(1, len(g0_v))) if len(g0_v) else 0.0
        g1a = float((g1_v == 1).sum() / max(1, len(g1_v))) if len(g1_v) else 0.0
        stats.append({"id": col, "agree": agree, "g0": g0a, "g1": g1a, "gap": g0a - g1a})

    honest_ranked = sorted(stats, key=lambda x: -x["agree"])
    g0_ranked = sorted(stats, key=lambda x: -x["g0"])
    g1_ranked = sorted(stats, key=lambda x: -x["g1"])

    ids = lambda lst: set(s["id"] for s in lst[:top_k])
    honest_set, g0_set, g1_set = ids(honest_ranked), ids(g0_ranked), ids(g1_ranked)

    # Strategic voting
    contentious = [s for s in stats if abs(s["gap"]) > gap_threshold]
    V_manip = V_raw.copy()
    for s in contentious:
        c = list(vote_cols).index(s["id"])
        for i in g0_idx:
            V_manip[i, c] = 1.0 if s["gap"] > 0 else -1.0

    # Use same column set as honest stats (only cols with ≥ 10 votes)
    valid_ids = set(s["id"] for s in stats)
    manip_stats = []
    for idx, col in enumerate(vote_cols):
        if col not in valid_ids:
            continue
        vm = V_manip[:, idx]
        valid_m = ~np.isnan(vm)
        am = float((vm[valid_m] == 1).sum() / max(1, valid_m.sum()))
        manip_stats.append({"id": col, "agree_m": am})

    manip_ranked = sorted(manip_stats, key=lambda x: -x["agree_m"])
    manip_set = ids(manip_ranked)

    n_entered = len(manip_set - honest_set)
    n_exited = len(honest_set - manip_set)

    return ConstitutionResult(
        n_contentious=len(contentious),
        J_honest_g0=jaccard(honest_set, g0_set),
        J_honest_g1=jaccard(honest_set, g1_set),
        J_g0_g1=jaccard(g0_set, g1_set),
        J_honest_manip=jaccard(honest_set, manip_set),
        n_entered=n_entered,
        n_exited=n_exited,
    )
