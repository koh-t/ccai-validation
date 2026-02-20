"""
Data loading for CCAI Polis dataset.

Expected files in *data_dir*:
  - clean_votes.csv   (required)
  - clean_comments.csv (optional, for statement text)

Download from https://github.com/saffronh/ccai
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .core import Mat, impute_votes

META_COLS = [
    "participant", "xid", "group-id",
    "n-comments", "n-votes", "n-agree", "n-disagree",
]


@dataclass
class CCAIData:
    """Container for loaded CCAI data."""

    V_raw: Mat
    """(n, d) raw vote matrix with NaN for missing."""

    V_filled: Mat
    """(n, d) imputed vote matrix (NaN → 0)."""

    groups: NDArray[np.intp]
    """(n,) Polis group-id per agent (0 or 1)."""

    vote_cols: list[str]
    """Column names (statement ids as strings)."""

    comments_df: pd.DataFrame | None = field(default=None)
    """Optional DataFrame with statement text."""

    @property
    def n(self) -> int:
        return self.V_raw.shape[0]

    @property
    def d(self) -> int:
        return self.V_raw.shape[1]

    @property
    def g0_idx(self) -> NDArray[np.intp]:
        return np.where(self.groups == 0)[0]

    @property
    def g1_idx(self) -> NDArray[np.intp]:
        return np.where(self.groups == 1)[0]

    @property
    def sparsity(self) -> float:
        return float(np.isnan(self.V_raw).mean())


def load_ccai(data_dir: str) -> CCAIData:
    """
    Load and preprocess the CCAI Polis dataset.

    Raises
    ------
    FileNotFoundError
        If ``clean_votes.csv`` is not found in *data_dir*.
    """
    votes_path = os.path.join(data_dir, "clean_votes.csv")
    if not os.path.exists(votes_path):
        raise FileNotFoundError(
            f"{votes_path} not found.  "
            f"Download from https://github.com/saffronh/ccai"
        )

    votes_df = pd.read_csv(votes_path)
    vote_cols = [c for c in votes_df.columns if c not in META_COLS]
    groups = votes_df["group-id"].values.astype(np.intp)
    V_raw = votes_df[vote_cols].values.astype(float)
    V_filled = impute_votes(V_raw, fill=0.0)

    comments_path = os.path.join(data_dir, "clean_comments.csv")
    comments_df = (
        pd.read_csv(comments_path) if os.path.exists(comments_path) else None
    )

    return CCAIData(
        V_raw=V_raw,
        V_filled=V_filled,
        groups=groups,
        vote_cols=vote_cols,
        comments_df=comments_df,
    )
