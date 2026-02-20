"""Shared fixtures for unit tests (no real CCAI data needed)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def triangle_ideals():
    """
    Example 4.10: three agents near distinct vertices of Δ².
    p1=(0.8,0.1,0.1), p2=(0.1,0.8,0.1), p3=(0.1,0.1,0.8).
    """
    return np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])


@pytest.fixture
def triangle_groups():
    """Groups for 3 agents: first two in G0, third in G1."""
    return np.array([0, 0, 1], dtype=np.intp)


@pytest.fixture
def symmetric_5d(rng):
    """50 agents on Δ⁴ from Dirichlet(1,...,1)."""
    raw = rng.dirichlet(np.ones(5), size=50)
    return raw


@pytest.fixture
def groups_50():
    """30 G0 + 20 G1."""
    return np.array([0] * 30 + [1] * 20, dtype=np.intp)


@pytest.fixture
def large_ideals(rng):
    """200 agents on Δ⁹ from Dirichlet(2,...,2)."""
    return rng.dirichlet(2 * np.ones(10), size=200)


@pytest.fixture
def large_groups():
    """120 G0 (60%) + 80 G1 (40%)."""
    return np.array([0] * 120 + [1] * 80, dtype=np.intp)


@pytest.fixture
def vote_matrix(rng):
    """Synthetic 100×30 vote matrix with NaN sparsity."""
    V = rng.choice([-1.0, 0.0, 1.0], size=(100, 30), p=[0.3, 0.1, 0.6])
    # introduce 40% NaN
    mask = rng.random((100, 30)) < 0.4
    V[mask] = np.nan
    return V


@pytest.fixture
def vote_cols_30():
    return [str(i) for i in range(30)]


@pytest.fixture
def groups_100():
    return np.array([0] * 60 + [1] * 40, dtype=np.intp)
