"""Tests for OVL computation and block-aware aggregation."""

import numpy as np
import pytest

from coop_stat.config import DimerSystem
from coop_stat.ovl.core import ovl_from_counts, ovl_from_hist_h5
from coop_stat.ovl.aggregation import (
    bucket_dataset_name,
    build_means_dict,
    compute_block_aware_mean,
)


class TestOvlFromCounts:
    """Unit tests for the core OVL calculation."""

    def test_identical_distributions(self):
        """OVL of identical distributions = 1.0."""
        counts = np.array([[[10, 20, 30]]], dtype=float)
        result = ovl_from_counts(counts, counts)
        np.testing.assert_allclose(result, [[1.0]], atol=1e-6)

    def test_disjoint_distributions(self):
        """OVL of non-overlapping distributions = 0.0."""
        cA = np.array([[[100, 0, 0]]], dtype=float)
        cB = np.array([[[0, 0, 100]]], dtype=float)
        result = ovl_from_counts(cA, cB)
        np.testing.assert_allclose(result, [[0.0]], atol=1e-6)

    def test_partial_overlap(self):
        """OVL should be between 0 and 1."""
        rng = np.random.default_rng(42)
        n, bins = 5, 10
        cA = rng.poisson(50, (n, n, bins)).astype(float)
        cB = rng.poisson(50, (n, n, bins)).astype(float)
        result = ovl_from_counts(cA, cB)
        assert result.shape == (n, n)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetric(self):
        """OVL(A, B) == OVL(B, A)."""
        rng = np.random.default_rng(42)
        cA = rng.poisson(50, (3, 3, 8)).astype(float)
        cB = rng.poisson(50, (3, 3, 8)).astype(float)
        np.testing.assert_array_equal(
            ovl_from_counts(cA, cB),
            ovl_from_counts(cB, cA),
        )

    def test_empty_bins(self):
        """Zero counts → OVL = 0."""
        zeros = np.zeros((2, 2, 5), dtype=float)
        result = ovl_from_counts(zeros, zeros)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            ovl_from_counts(np.zeros((2, 2, 5)), np.zeros((3, 3, 5)))


class TestOvlFromHistH5:
    """Integration test with actual H5 files."""

    def test_basic(self, hist_h5_pair):
        pathA, pathB = hist_h5_pair
        result = ovl_from_hist_h5(pathA, pathB)
        assert result.ndim == 2
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_self_overlap(self, hist_h5_pair):
        """OVL of file with itself should be all 1.0."""
        pathA, _ = hist_h5_pair
        result = ovl_from_hist_h5(pathA, pathA)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)


class TestBlockAwareMean:
    """Tests for per-bucket DD-deduped mean computation."""

    def _make_records(self, dimer_n=10, matrix_n=15, n_reps=3, base=0.5, seed=42):
        """Build fake comparison records."""
        rng = np.random.default_rng(seed)
        records = []
        for ra in range(1, n_reps + 1):
            for rb in range(ra + 1, n_reps + 1):
                ovl = np.clip(
                    base + rng.normal(0, 0.05, (matrix_n, matrix_n)),
                    0, 1,
                ).astype(np.float32)
                # Make DD block identical across views for same rep-pair
                dd_block = ovl[:dimer_n, :dimer_n].copy()
                for view in ["PLmat1", "PLmat2as1"]:
                    rec_ovl = ovl.copy()
                    rec_ovl[:dimer_n, :dimer_n] = dd_block  # same DD
                    records.append({
                        "ovl": rec_ovl,
                        "rep_A": ra,
                        "rep_B": rb,
                        "view_A": view,
                        "view_B": view,
                        "client_A": "BSN2",
                        "client_B": "BSN2",
                        "view_pair_label": f"{view}(BSN2)-vs-{view}(BSN2)",
                        "key": f"test_r{ra}_{view}_vs_r{rb}_{view}",
                    })
        return records

    def test_basic(self, tiny_system):
        records = self._make_records(dimer_n=tiny_system.dimer_n, matrix_n=15)
        means, unblocked, audit = compute_block_aware_mean(
            records, tiny_system, matrix_n=15
        )
        assert len(means) > 0
        assert unblocked.shape == (15, 15)
        assert audit["total_records"] == len(records)

    def test_dd_integrity_check(self, tiny_system):
        """If DD blocks differ for same rep-pair, should raise."""
        records = self._make_records(dimer_n=tiny_system.dimer_n, matrix_n=15)
        # Corrupt one DD block
        records[1]["ovl"][:tiny_system.dimer_n, :tiny_system.dimer_n] += 1.0
        with pytest.raises(ValueError, match="DD block mismatch"):
            compute_block_aware_mean(records, tiny_system, matrix_n=15)

    def test_no_records(self, tiny_system):
        means, unblocked, audit = compute_block_aware_mean([], tiny_system, matrix_n=15)
        assert len(means) == 0
        np.testing.assert_array_equal(unblocked, np.zeros((15, 15)))


class TestBucketDatasetName:
    def test_same_client(self):
        assert bucket_dataset_name("PLmat1", "BSN2", "BSN2") == "ovl_mean_deduped_PLmat1_BSN2"

    def test_cross_client(self):
        assert bucket_dataset_name("native", "BSN2", "ICE1") == "ovl_mean_deduped_native_BSN2_x_ICE1"


class TestBuildMeansDict:
    def test_keys(self):
        means = {("PLmat1", "BSN2", "BSN2"): np.zeros((5, 5))}
        unblocked = np.zeros((5, 5))
        d = build_means_dict(means, unblocked)
        assert "ovl_mean_deduped_PLmat1_BSN2" in d
        assert "ovl_mean_eligible_unblocked" in d
