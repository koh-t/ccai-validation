"""
CLI entry point.

Usage:
    uv run ccai-run --data_dir ./data
    uv run ccai-run --data_dir ./data --experiments pac,pairwise
    uv run ccai-run --data_dir ./data --dims 3,5,10
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .core import embed_simplex
from .data import load_ccai
from .experiments import (
    exp_aggregation,
    exp_constitution,
    exp_mechanism_properties,
    exp_pac,
    exp_pairwise,
    exp_sharding,
)

ALL_EXPERIMENTS = ["pac", "pairwise", "aggregation", "mechanism", "sharding", "constitution"]


def _header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def run_pac(V_filled, groups, dims):
    _header("TABLE 1: Partial Agreement Cone (Lemma 4.3)")
    print(f"  {'m':>3} {'PAC':>8} {'G0':>8} {'G1':>8} {'Cross':>8} {'Anti':>8} {'cosθ':>8}")
    print(f"  {'-' * 55}")
    for m in dims:
        ideals, _ = embed_simplex(V_filled, m)
        r = exp_pac(ideals, groups)
        print(
            f"  {m:>3} {r.rates['all']:>7.1%} {r.rates['intra_g0']:>7.1%} "
            f"{r.rates['intra_g1']:>7.1%} {r.rates['cross']:>7.1%} "
            f"{r.antiparallel_frac:>7.1%} {r.mean_cos:>+7.4f}"
        )


def run_pairwise(V_filled, groups, dims):
    _header("TABLE 2: Pairwise Logrolling (Example 4.10)")
    print(f"  {'m':>3} {'success':>8} {'/500':>6} {'rate':>7} {'gain':>10}")
    print(f"  {'-' * 38}")
    for m in dims:
        ideals, _ = embed_simplex(V_filled, m)
        r = exp_pairwise(ideals)
        mg = np.mean([g[0] + g[1] for g in r.gains]) if r.gains else 0
        print(f"  {m:>3} {r.success:>8} {'/' + str(r.total):>6} {r.rate:>6.1%} {mg:>10.1e}")


def run_aggregation(V_filled, groups):
    _header("TABLE 3: Aggregation Rules (s=0.01, m=5)")
    ideals, _ = embed_simplex(V_filled, 5)
    g0_n = int((groups == 0).sum())
    print(f"  |S|={g0_n} ({g0_n / len(groups):.0%})")
    print(f"  {'Rule':<22} {'Shift':>10} {'G0 Δu':>10} {'G0%gain':>8}")
    print(f"  {'-' * 52}")
    for r in exp_aggregation(ideals, groups):
        std = f" ± {r.shift_std:.1e}" if r.shift_std else ""
        print(f"  {r.rule:<22} {r.shift:>9.1e}{std} {r.g0_du:>+9.1e} {r.g0_frac:>7.1%}")


def run_mechanism(V_filled, groups, dims):
    _header("TABLE 4: Per-Axis SP + Panel Independence")
    print(f"  {'m':>3} {'SP viol':>12} {'‖ΔAxes‖':>12}")
    print(f"  {'-' * 30}")
    for m in dims:
        ideals, _ = embed_simplex(V_filled, m)
        r = exp_mechanism_properties(ideals, groups)
        print(f"  {m:>3} {r.sp_violation:>11.4%} {r.panel_independence:>12.1e}")


def run_sharding(V_filled, groups, dims):
    _header("TABLE 5a: Sharding — Majority Coalition")
    for m in dims:
        ideals, _ = embed_simplex(V_filled, m)
        r = exp_sharding(ideals, groups, use_minority=False)
        K = m - 1
        print(f"\n  m={m}, K={K}, ρ={r.rho:.2f}")
        for lab, s, g in [
            ("Sharded", r.shifts_sharded, r.gains_sharded),
            ("Unsharded", r.shifts_unsharded, r.gains_unsharded),
        ]:
            print(f"  {lab:<12} shift={np.mean(s):.1e}±{np.std(s):.1e}  G0%={np.mean(g):.1%}")
        if r.captures:
            c = np.array(r.captures)
            print(f"  Captured: {c.mean():.1f}/{K}, P(≥2)={float((c >= 2).mean()):.3f}")

    _header("TABLE 5b: Sharding — Minority Coalition (ρ=0.30)")
    for m in dims:
        ideals, _ = embed_simplex(V_filled, m)
        r = exp_sharding(ideals, groups, use_minority=True, minority_frac=0.30)
        K = m - 1
        b = (len(groups) // 2) // K
        print(f"\n  m={m}, K={K}, ρ={r.rho:.2f}, b≈{b}")
        for lab, s, g in [
            ("Sharded", r.shifts_sharded, r.gains_sharded),
            ("Unsharded", r.shifts_unsharded, r.gains_unsharded),
        ]:
            print(f"  {lab:<12} shift={np.mean(s):.1e}±{np.std(s):.1e}  G0%={np.mean(g):.1%}")
        if r.captures:
            c = np.array(r.captures)
            print(f"  Captured: {c.mean():.1f}/{K}, P(≥2)={float((c >= 2).mean()):.3f}")
            p1 = np.exp(-2 * b * (0.5 - r.rho) ** 2)
            print(f"  Hoeffding: P(1)≤{p1:.2e}, E[C]≤{K * p1:.2e}")


def run_constitution(V_raw, groups, vote_cols):
    _header("TABLE 6: Constitution-Level Manipulation")
    r = exp_constitution(V_raw, groups, vote_cols)
    print(f"  Contentious (|gap|>5%): {r.n_contentious}")
    print(f"  J(Honest, G0)    = {r.J_honest_g0:.2f}")
    print(f"  J(Honest, G1)    = {r.J_honest_g1:.2f}")
    print(f"  J(G0, G1)        = {r.J_g0_g1:.2f}")
    print(f"  J(Honest, Manip) = {r.J_honest_manip:.2f}")
    print(f"  Entered: {r.n_entered}, Exited: {r.n_exited} / 15")


def main():
    parser = argparse.ArgumentParser(
        description="Empirical validation of Sharded Citizens' Assembly"
    )
    parser.add_argument("--data_dir", default="./data",
                        help="Path to CCAI data directory")
    parser.add_argument("--experiments", default=",".join(ALL_EXPERIMENTS),
                        help=f"Comma-separated experiments: {ALL_EXPERIMENTS}")
    parser.add_argument("--dims", default="3,5,10",
                        help="Comma-separated simplex dimensions")
    args = parser.parse_args()

    exps = [e.strip() for e in args.experiments.split(",")]
    dims = [int(d.strip()) for d in args.dims.split(",")]

    print("=" * 70)
    print("  EMPIRICAL VALIDATION — SHARDED CITIZENS' ASSEMBLY ON CCAI DATA")
    print("=" * 70)

    data = load_ccai(args.data_dir)
    print(f"  {data.n} agents × {data.d} statements, sparsity={data.sparsity:.1%}")
    print(f"  G0={len(data.g0_idx)}, G1={len(data.g1_idx)}")

    dispatch = {
        "pac":          lambda: run_pac(data.V_filled, data.groups, dims),
        "pairwise":     lambda: run_pairwise(data.V_filled, data.groups, dims),
        "aggregation":  lambda: run_aggregation(data.V_filled, data.groups),
        "mechanism":    lambda: run_mechanism(data.V_filled, data.groups, dims),
        "sharding":     lambda: run_sharding(data.V_filled, data.groups, dims),
        "constitution": lambda: run_constitution(data.V_raw, data.groups, data.vote_cols),
    }

    for exp in exps:
        if exp not in dispatch:
            print(f"  [WARN] Unknown experiment: {exp}", file=sys.stderr)
            continue
        dispatch[exp]()

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")
