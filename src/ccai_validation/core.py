"""
Core mathematical primitives.

All functions operate on ideal points in Δ^{m-1} (the open simplex)
and use ℓ₂ utility: u_i(π) = -‖π - p_i‖².
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from scipy.stats import trim_mean
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Type alias
Vec = NDArray[np.floating]
Mat = NDArray[np.floating]

EPS = 1e-15


# ------------------------------------------------------------------
# Simplex operations
# ------------------------------------------------------------------

def to_simplex(x: Vec) -> Vec:
    """Project a positive vector onto the probability simplex by normalising."""
    x = np.maximum(x, EPS)
    return x / x.sum()


def is_simplex(x: Vec, atol: float = 1e-8) -> bool:
    """Check whether *x* lies in Δ^{m-1}."""
    return bool(np.all(x > -atol) and abs(x.sum() - 1.0) < atol)


# ------------------------------------------------------------------
# Embedding: vote matrix → ideal points on Δ^{m-1}
# ------------------------------------------------------------------

def embed_simplex(V_filled: Mat, m: int) -> tuple[Mat, PCA]:
    """
    PCA + softmax embedding.

    Parameters
    ----------
    V_filled : (n, d) vote matrix (NaN already imputed).
    m        : target simplex dimension (number of alternatives).

    Returns
    -------
    ideals : (n, m) array on Δ^{m-1}.
    pca    : fitted PCA object.
    """
    pca = PCA(n_components=m)
    scores = pca.fit_transform(V_filled)
    ideals = softmax(scores, axis=1)
    return ideals, pca


def impute_votes(V_raw: Mat, fill: float = 0.0) -> Mat:
    """Replace NaN with *fill* (default 0 = "pass")."""
    imp = SimpleImputer(strategy="constant", fill_value=fill)
    return imp.fit_transform(V_raw)


# ------------------------------------------------------------------
# ℓ₂ utility and gradient (Equations 1–2 in paper)
# ------------------------------------------------------------------

def l2_sq(a: Vec, b: Vec) -> float:
    """‖a − b‖²."""
    return float(np.sum((a - b) ** 2))


def l2_utility(pi: Vec, p_i: Vec) -> float:
    """u_i(π) = −‖π − p_i‖²."""
    return -l2_sq(pi, p_i)


def l2_gradient(pi: Vec, p_i: Vec) -> Vec:
    """∇_π u_i(π) = 2(p_i − π).  Lies in T∆ since Σ(p_i − π) = 0."""
    return 2.0 * (p_i - pi)


# ------------------------------------------------------------------
# Aggregation rules (Section 6.4 / Table 1)
# ------------------------------------------------------------------

def cw_median(ideals: Mat) -> Vec:
    """Coordinate-wise median, renormalised to Δ."""
    return to_simplex(np.median(ideals, axis=0))


def cw_mean(ideals: Mat) -> Vec:
    """Coordinate-wise mean, renormalised to Δ."""
    return to_simplex(ideals.mean(axis=0))


def cw_trimmed_mean(ideals: Mat, trim: float = 0.1) -> Vec:
    """Coordinate-wise trimmed mean (symmetric *trim* fraction), renormalised."""
    m = ideals.shape[1]
    tm = np.array([trim_mean(ideals[:, k], trim) for k in range(m)])
    return to_simplex(tm)


# ------------------------------------------------------------------
# Partial-agreement cone (Definition 4.1)
# ------------------------------------------------------------------

def pac_nonempty(gi: Vec, gj: Vec) -> bool:
    """
    Return True iff the partial agreement cone C_{ij} is nonempty.

    C_{ij} = {d ∈ T∆ : ⟨d, g_i⟩ > 0 and ⟨d, g_j⟩ > 0}.
    Nonempty ⟺ g_i and g_j are not antiparallel.
    """
    ni = np.linalg.norm(gi)
    nj = np.linalg.norm(gj)
    if ni < EPS or nj < EPS:
        return False
    cos_ij = np.dot(gi, gj) / (ni * nj)
    # Antiparallel means cos = −1 exactly; any cos > −1 ⇒ nonempty
    return bool(cos_ij > -1 + 1e-10)


def gradient_cos(gi: Vec, gj: Vec) -> float:
    """Cosine of angle between two gradient vectors."""
    ni = np.linalg.norm(gi)
    nj = np.linalg.norm(gj)
    if ni < EPS or nj < EPS:
        return 0.0
    return float(np.dot(gi, gj) / (ni * nj))


# ------------------------------------------------------------------
# Coalition manipulation helpers
# ------------------------------------------------------------------

def coalition_target(
    ideals: Mat,
    coalition_idx: NDArray[np.intp],
    pi_star: Vec,
    strength: float = 0.01,
) -> Vec:
    """
    Compute the coalition's optimal target point π* + s·d*.

    d* is the mean ℓ₂ gradient of coalition members, projected to T∆
    and normalised.
    """
    if len(coalition_idx) == 0:
        return pi_star.copy()
    grads = np.array([l2_gradient(pi_star, ideals[i]) for i in coalition_idx])
    d_star = grads.mean(axis=0)
    d_star -= d_star.mean()  # project to T∆
    norm = np.linalg.norm(d_star)
    if norm < EPS:
        return pi_star.copy()
    d_star /= norm
    target = pi_star + strength * d_star
    return to_simplex(target)


def utility_gain(
    pi_new: Vec, pi_old: Vec, ideals: Mat, idx: NDArray[np.intp]
) -> tuple[float, float]:
    """
    Mean Δu and fraction with positive gain for agents in *idx*.

    Returns (mean_du, frac_gain).
    """
    if len(idx) == 0:
        return 0.0, 0.0
    du = np.array([l2_utility(pi_new, ideals[i]) - l2_utility(pi_old, ideals[i])
                   for i in idx])
    return float(du.mean()), float((du > 0).mean())


def jaccard(a: set, b: set) -> float:
    """Jaccard index |A∩B| / |A∪B|."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
