"""Tests for OVL value extraction and KS statistics."""

import numpy as np
import pytest

from dimer_ovl.compare.extraction import (
    client_from_view_obj,
    client_match,
    extract_values_for_pair,
    is_dimer_key,
    is_intra_same_rep,
    parse_pair_key,
    rep_pair_from_key,
    smart_mean,
)
from dimer_ovl.compare.statistics import ks_pvalue, compute_pair_stats
from dimer_ovl.config import DimerSystem


# ── Key parsing ──────────────────────────────────────────────────────


class TestKeyParsing:
    def test_is_dimer_key(self):
        assert is_dimer_key("BSN2d15_r1_dimer_vs_BSN2d15_r2_dimer")
        assert not is_dimer_key("BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2")

    def test_parse_pair_key(self):
        key = "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2"
        result = parse_pair_key(key)
        assert result is not None
        Lt, Lo, Rt, Ro = result
        assert Lt == "BSN2d15"
        assert Rt == "BSN2d15"

    def test_rep_pair_from_key(self):
        key = "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r3_PLmat1_BSN2"
        ra, rb = rep_pair_from_key(key)
        assert ra == 1
        assert rb == 3

    def test_is_intra_same_rep(self):
        same = "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r1_PLmat1_BSN2"
        diff = "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2"
        assert is_intra_same_rep(same)
        assert not is_intra_same_rep(diff)


class TestClientMatch:
    def test_no_filter(self):
        assert client_match("anything", None)

    def test_dimer_key_always_passes(self):
        assert client_match("r1_dimer_vs_r2_dimer", "BSN2")

    def test_client_from_view_obj(self):
        assert client_from_view_obj("r1_PLmat1_BSN2") == "BSN2"
        assert client_from_view_obj("r2_PLmat2as1_SPAG5") == "SPAG5"
        assert client_from_view_obj("r1_native_ICE1") == "ICE1"
        assert client_from_view_obj("some_random_string") is None


# ── Value extraction ─────────────────────────────────────────────────


class TestExtractValues:
    @pytest.fixture
    def small_system(self):
        return DimerSystem(chain_length=5)

    def test_empty_dict(self, small_system):
        assert extract_values_for_pair(None, 0, 1, small_system) == []
        assert extract_values_for_pair({}, 0, 1, small_system) == []

    def test_dd_pair_dedup(self, small_system):
        """DD pairs should deduplicate by rep-pair."""
        M1 = np.full((10, 10), 0.5, dtype=np.float32)
        M2 = np.full((10, 10), 0.6, dtype=np.float32)
        ovl_dict = {
            "TAG_r1_PLmat1_X_vs_TAG_r2_PLmat1_X": M1,
            "TAG_r1_dimer_vs_TAG_r2_dimer": M1[:10, :10],
        }
        vals = extract_values_for_pair(ovl_dict, 0, 1, small_system)
        # Should get exactly 1 value (one rep-pair)
        assert len(vals) == 1
        assert vals[0] == pytest.approx(0.5)

    def test_dd_sym_expansion(self, small_system):
        """With dd_sym=True, should get both direct and sym values."""
        M = np.random.default_rng(42).random((10, 10)).astype(np.float32)
        M = (M + M.T) / 2
        ovl_dict = {
            "TAG_r1_PLmat1_X_vs_TAG_r2_PLmat1_X": M,
            "TAG_r1_dimer_vs_TAG_r2_dimer": M,
        }
        # Pair (0, 1) → sym pair (5, 6) for chain_length=5
        vals_no_sym = extract_values_for_pair(ovl_dict, 0, 1, small_system, dd_sym=False)
        vals_sym = extract_values_for_pair(ovl_dict, 0, 1, small_system, dd_sym=True)
        assert len(vals_no_sym) == 1
        assert len(vals_sym) == 2  # direct + sym

    def test_same_rep_excluded(self, small_system):
        """Same-rep intra keys should be excluded."""
        M = np.full((10, 10), 0.9, dtype=np.float32)
        ovl_dict = {
            "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r1_PLmat1_BSN2": M,  # same rep
            "BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2": M,  # diff rep
        }
        vals = extract_values_for_pair(ovl_dict, 0, 1, small_system)
        assert len(vals) == 1  # only the cross-rep pair


class TestSmartMean:
    def test_excludes_dimer_keys(self):
        M_view = np.full((5, 5), 0.5, dtype=np.float32)
        M_dimer = np.full((5, 5), 0.9, dtype=np.float32)
        d = {
            "TAG_r1_PLmat1_X_vs_TAG_r2_PLmat1_X": M_view,
            "TAG_r1_dimer_vs_TAG_r2_dimer": M_dimer,
        }
        result = smart_mean(d)
        assert result is not None
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_empty(self):
        assert smart_mean(None) is None
        assert smart_mean({}) is None


# ── KS statistics ────────────────────────────────────────────────────


class TestKsPvalue:
    def test_identical_samples(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = ks_pvalue(a, a)
        assert p is not None
        assert p == 1.0  # identical → p=1

    def test_different_samples(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(10, 1, 50)
        p = ks_pvalue(a, b)
        assert p is not None
        assert p < 0.05  # very different

    def test_too_few_samples(self):
        assert ks_pvalue(np.array([1.0]), np.array([2.0, 3.0])) is None
        assert ks_pvalue(np.array([1.0, 2.0]), np.array([3.0])) is None

    def test_mc_method(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 50)
        b = rng.normal(5, 1, 50)
        p = ks_pvalue(a, b, method="mc", mc_n=1000, rng=rng)
        assert p is not None
        assert p < 0.05


class TestComputePairStats:
    def test_basic(self):
        system = DimerSystem(chain_length=5)
        M_cross = np.full((10, 10), 0.4, dtype=np.float32)
        M_intra = np.full((10, 10), 0.8, dtype=np.float32)
        cross = {"TAG_r1_PLmat1_X_vs_TAG_r2_PLmat1_X": M_cross}
        intra = {"TAG_r1_PLmat1_X_vs_TAG_r2_PLmat1_X": M_intra}

        stats = compute_pair_stats(
            0, 1, cross, intra, intra,
            system=system, nres=10,
            dd_sym_cross=False, dd_sym_iref=False, dd_sym_icmp=False,
            matched_client=None,
        )
        assert stats["cm"] is not None
        assert stats["rm"] is not None
        assert stats["cm"] < stats["rm"]  # cross < intra
