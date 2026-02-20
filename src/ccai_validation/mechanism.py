"""
Sharded Citizens' Assembly mechanism (Definition 5.1).

Steps:
  1. Partition agents into Panel A (axis extraction) and Panel B (voting).
  2. PCA on Panel A reports → K orthonormal axes.
  3. Shard Panel B into K subpanels; each subpanel votes on one axis by median.
  4. Reconstruct: μ_A + Σ_k median_k · e_k, then project to Δ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from .core import EPS, Mat, Vec, to_simplex

# Type alias for subpanel indices (local indices within B_idx)
Subpanels = list[NDArray[np.intp]]


@dataclass
class CAResult:
    """Return value of :func:`citizens_assembly`."""

    pi: Vec
    """Aggregated policy on Δ^{m-1}."""

    A_idx: NDArray[np.intp]
    """Global indices of Panel-A agents."""

    B_idx: NDArray[np.intp]
    """Global indices of Panel-B agents."""

    axes: Mat
    """(K, m) orthonormal axis matrix extracted from Panel A."""

    mu_A: Vec
    """Panel-A centroid."""

    projections: Mat
    """(|B|, K) projections of Panel-B agents onto axes."""

    medians: Vec
    """(K,) per-axis medians."""

    subpanels: Subpanels | None = field(default=None)
    """
    List of K arrays, each holding local indices (within B_idx) of the
    agents in that subpanel.  ``None`` when ``shard=False``.
    """


def citizens_assembly(
    ideals: Mat,
    *,
    K: int | None = None,
    panel_a_frac: float = 0.5,
    seed: int = 42,
    shard: bool = True,
) -> CAResult:
    """
    Run the Sharded Citizens' Assembly.

    Parameters
    ----------
    ideals : (n, m) ideal points on Δ^{m-1}.
    K      : number of axes to extract (default ``min(m-1, 5)``).
    panel_a_frac : fraction of agents assigned to Panel A.
    seed   : random seed (controls partition & shard assignment).
    shard  : if True, Panel B is sharded into K disjoint subpanels.

    Returns
    -------
    CAResult with aggregated policy and diagnostic fields.
    """
    rng = np.random.RandomState(seed)
    n, m = ideals.shape
    if K is None:
        K = min(m - 1, 5)

    # Step 1: Partition
    idx = rng.permutation(n)
    n_A = int(n * panel_a_frac)
    A_idx: NDArray[np.intp] = idx[:n_A]
    B_idx: NDArray[np.intp] = idx[n_A:]

    # Step 2: Axis extraction (Panel A)
    mu_A = ideals[A_idx].mean(axis=0)
    centered_A = ideals[A_idx] - mu_A
    pca_A = PCA(n_components=K)
    pca_A.fit(centered_A)
    axes = pca_A.components_  # (K, m)

    # Step 3: Voting (Panel B)
    centered_B = ideals[B_idx] - mu_A
    projections = centered_B @ axes.T  # (n_B, K)

    subpanels: Subpanels | None
    medians = np.zeros(K)

    if shard:
        B_perm = rng.permutation(len(B_idx))
        panel_size = len(B_idx) // K
        subpanels = []
        for k in range(K):
            start = k * panel_size
            end = (k + 1) * panel_size if k < K - 1 else len(B_idx)
            panel_k = B_perm[start:end]
            subpanels.append(panel_k)
            medians[k] = np.median(projections[panel_k, k])
    else:
        medians = np.median(projections, axis=0)
        subpanels = None

    # Step 4: Reconstruction
    pi_hat = mu_A + medians @ axes
    pi_star = to_simplex(pi_hat)

    return CAResult(
        pi=pi_star,
        A_idx=A_idx,
        B_idx=B_idx,
        axes=axes,
        mu_A=mu_A,
        projections=projections,
        medians=medians,
        subpanels=subpanels,
    )
