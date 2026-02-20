"""Tests for ccai_validation.data."""

import numpy as np
import os
import pytest

from ccai_validation.data import CCAIData, load_ccai


class TestCCAIDataProperties:
    """Test CCAIData dataclass computed properties."""

    def test_n_and_d(self):
        V = np.zeros((10, 5))
        d = CCAIData(V_raw=V, V_filled=V,
                     groups=np.zeros(10, dtype=np.intp),
                     vote_cols=[str(i) for i in range(5)])
        assert d.n == 10
        assert d.d == 5

    def test_group_indices(self):
        groups = np.array([0, 0, 1, 1, 0], dtype=np.intp)
        d = CCAIData(V_raw=np.zeros((5, 2)), V_filled=np.zeros((5, 2)),
                     groups=groups, vote_cols=["a", "b"])
        assert list(d.g0_idx) == [0, 1, 4]
        assert list(d.g1_idx) == [2, 3]

    def test_sparsity(self):
        V = np.array([[1.0, np.nan], [np.nan, np.nan], [1.0, 1.0]])
        d = CCAIData(V_raw=V, V_filled=np.nan_to_num(V),
                     groups=np.zeros(3, dtype=np.intp),
                     vote_cols=["a", "b"])
        assert d.sparsity == pytest.approx(3 / 6)


class TestLoadCCAI:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="clean_votes.csv"):
            load_ccai(str(tmp_path))

    @pytest.mark.slow
    def test_real_data(self):
        """Requires actual CCAI data in ./ccai or /home/claude/ccai."""
        for path in ["./ccai", "/home/claude/ccai", "./data"]:
            if os.path.exists(os.path.join(path, "clean_votes.csv")):
                data = load_ccai(path)
                assert data.n > 900
                assert data.d > 200
                assert len(data.g0_idx) + len(data.g1_idx) == data.n
                assert not np.any(np.isnan(data.V_filled))
                return
        pytest.skip("CCAI data not found")
