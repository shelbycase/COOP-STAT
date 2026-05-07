"""Tests for histogram indexing and building."""

import h5py
import numpy as np
import pytest

from coop_stat.histogram.indexing import (
    build_arithmetic_index,
    build_resid_aware_index,
    canonical_size,
    identify_ligand_atoms,
)
from coop_stat.tags import parse_tag


class TestCanonicalSize:
    def test_singly_bound(self):
        tag = parse_tag("BSN2s15")
        assert canonical_size(tag, 193) == 193  # all atoms

    def test_doubly_bound(self):
        tag = parse_tag("BSN2d15")
        assert canonical_size(tag, 208) == 193  # 208 - 15


class TestIdentifyLigandAtoms:
    def test_simple_case(self):
        # 10 dimer (resids 1-10) + 5 ligA (11-15) + 5 ligB (16-20)
        resids = np.arange(1, 21)
        dimer, ligA, ligB = identify_ligand_atoms(resids, peplen=5)
        assert len(dimer) == 10
        assert len(ligA) == 5
        assert len(ligB) == 5
        # ligB should have higher resids
        assert resids[ligB[-1]] > resids[ligA[-1]]

    def test_shuffled_order(self):
        """Resids not in ascending raw order — should still work."""
        rng = np.random.default_rng(42)
        resids = np.arange(1, 21)
        order = rng.permutation(20)
        shuffled_resids = resids[order]

        dimer, ligA, ligB = identify_ligand_atoms(shuffled_resids, peplen=5)
        # Groups should be sorted ascending by resid
        assert all(shuffled_resids[ligA[i]] < shuffled_resids[ligA[i + 1]]
                    for i in range(len(ligA) - 1))
        assert all(shuffled_resids[ligB[i]] < shuffled_resids[ligB[i + 1]]
                    for i in range(len(ligB) - 1))

    def test_groups_are_disjoint(self):
        resids = np.arange(1, 21)
        dimer, ligA, ligB = identify_ligand_atoms(resids, peplen=5)
        all_idx = set(dimer.tolist()) | set(ligA.tolist()) | set(ligB.tolist())
        assert len(all_idx) == 20  # no overlap, no missing


class TestBuildResidAwareIndex:
    def test_singly_bound_sorts_by_resid(self):
        tag = parse_tag("BSN2s15")
        resids = np.array([5, 3, 1, 4, 2])
        idx, canon_resids, aware = build_resid_aware_index(tag, "native", 5, resids)
        assert aware
        np.testing.assert_array_equal(canon_resids, [1, 2, 3, 4, 5])

    def test_doubly_bound_site1(self):
        tag = parse_tag("AAAd5")
        resids = np.arange(1, 21)  # 10 dimer + 5 ligA + 5 ligB
        idx, canon_resids, _ = build_resid_aware_index(tag, "site1", 20, resids)
        assert len(idx) == 15  # dimer(10) + ligA(5)
        # Last 5 should be ligA (resids 11-15)
        np.testing.assert_array_equal(canon_resids[-5:], [11, 12, 13, 14, 15])

    def test_doubly_bound_site2as1(self):
        tag = parse_tag("AAAd5")
        resids = np.arange(1, 21)
        idx, canon_resids, _ = build_resid_aware_index(tag, "site2as1", 20, resids)
        assert len(idx) == 15
        # Last 5 should be ligB (resids 16-20)
        np.testing.assert_array_equal(canon_resids[-5:], [16, 17, 18, 19, 20])


class TestBuildArithmeticIndex:
    def test_singly_bound(self):
        tag = parse_tag("BSN2s15")
        idx = build_arithmetic_index(tag, "native", 193)
        assert len(idx) == 193
        np.testing.assert_array_equal(idx, np.arange(193))

    def test_doubly_site1(self):
        tag = parse_tag("BSN2d15")
        idx = build_arithmetic_index(tag, "site1", 208)
        assert len(idx) == 193


class TestEdgesAndHistogram:
    """Integration test: build edges then histogram from synthetic data."""

    def test_histogram_shape(self, hist_h5_pair):
        """Verify H5 files from fixture have expected structure."""
        pathA, pathB = hist_h5_pair
        with h5py.File(pathA, "r") as h5:
            counts = h5["hist_counts"][...]
            edges = h5["bin_edges"][...]
        assert counts.ndim == 3
        assert edges.shape[-1] == counts.shape[-1] + 1
