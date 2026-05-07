"""Tests for coop_stat.config (DimerSystem) and coop_stat.topology."""

import numpy as np
import pytest

from coop_stat.config import DimerSystem, lc8_system
from coop_stat.tags import parse_tag
from coop_stat.topology import (
    is_dimer_pair,
    matched_binding_window,
    matched_client_for_comparison,
    pair_equivalents,
)


class TestDimerSystem:
    """DimerSystem construction and properties."""

    def test_dimer_n(self, lc8):
        assert lc8.dimer_n == 178

    def test_chain_ranges(self, lc8):
        assert lc8.chain_a_range == (0, 89)
        assert lc8.chain_b_range == (89, 178)

    def test_tiny_system(self, tiny_system):
        assert tiny_system.dimer_n == 10
        assert tiny_system.chain_length == 5

    def test_invalid_chain_length(self):
        with pytest.raises(ValueError):
            DimerSystem(chain_length=0)

    def test_invalid_binding_window(self):
        with pytest.raises(ValueError, match="invalid"):
            DimerSystem(
                chain_length=5,
                binding_windows_1=[(10, 5)],  # lo > hi
            )


class TestC2Symmetry:
    """C2 mate and sym map."""

    def test_c2_mate_chain_a(self, lc8):
        assert lc8.c2_mate(0) == 89
        assert lc8.c2_mate(5) == 94
        assert lc8.c2_mate(88) == 177

    def test_c2_mate_chain_b(self, lc8):
        assert lc8.c2_mate(89) == 0
        assert lc8.c2_mate(177) == 88

    def test_c2_mate_out_of_range(self, lc8):
        assert lc8.c2_mate(-1) is None
        assert lc8.c2_mate(178) is None

    def test_c2_mate_involution(self, lc8):
        """c2_mate(c2_mate(i)) == i for all valid indices."""
        for i in range(lc8.dimer_n):
            assert lc8.c2_mate(lc8.c2_mate(i)) == i

    def test_c2_sym_map_shape(self, lc8):
        sm = lc8.c2_sym_map()
        assert sm.shape == (178,)
        assert sm.dtype == np.int32

    def test_c2_sym_map_values(self, tiny_system):
        sm = tiny_system.c2_sym_map()
        np.testing.assert_array_equal(sm, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4])

    def test_c2_sym_map_involution(self, lc8):
        sm = lc8.c2_sym_map()
        np.testing.assert_array_equal(sm[sm], np.arange(178))


class TestBindingWindows:
    def test_bw1(self, lc8):
        idx = lc8.binding_indices(bw=1)
        assert 1 in idx
        assert 24 in idx
        assert 25 not in idx

    def test_bw2(self, lc8):
        idx = lc8.binding_indices(bw=2)
        assert 25 in idx
        assert 58 in idx
        assert 24 not in idx

    def test_bw_both(self, lc8):
        idx = lc8.binding_indices(bw=0)
        assert 1 in idx and 25 in idx

    def test_residue_bw(self, lc8):
        assert lc8.residue_binding_window(5) == 1
        assert lc8.residue_binding_window(30) == 2
        assert lc8.residue_binding_window(0) == 0  # outside both


class TestClientIdentification:
    def test_known_client(self, lc8):
        assert lc8.identify_client("YPRATAEFSTQTPSP") == "BSN2"
        assert lc8.identify_client("YHPETQDSSTQTDTS") == "SPAG5"

    def test_unknown_client(self, lc8):
        with pytest.raises(ValueError, match="Cannot match"):
            lc8.identify_client("XXXXXXXXXXX")


class TestPairEquivalents:
    def test_direct_only(self, lc8):
        eq = pair_equivalents(5, 10, lc8, expand_sym=False)
        assert len(eq) == 1
        assert eq[0] == (5, 10, "direct")

    def test_with_sym(self, lc8):
        eq = pair_equivalents(5, 10, lc8, expand_sym=True)
        assert len(eq) == 2
        assert eq[0] == (5, 10, "direct")
        assert eq[1] == (94, 99, "sym")

    def test_cross_chain_no_sym(self, lc8):
        """Pair spanning both chains — sym pair is the same pair (reversed)."""
        eq = pair_equivalents(5, 94, lc8, expand_sym=True)
        # (5, 94) and its C2 partner are (5, 94) again
        assert len(eq) == 1  # no new pair

    def test_is_dimer_pair(self, lc8):
        assert is_dimer_pair(5, 10, lc8)
        assert is_dimer_pair(5, 94, lc8)
        assert not is_dimer_pair(5, 200, lc8)  # beyond dimer_n


class TestMatchedBindingWindow:
    def test_d_homo_vs_s(self, lc8):
        tr = parse_tag("BSN2d15")
        tc = parse_tag("BSN2s15")
        assert matched_binding_window(tr, tc, lc8) == 0

    def test_d_het_vs_d_het_core1_match(self, lc8):
        tr = parse_tag("BSN2m15_SPAG5m15")
        tc = parse_tag("BSN2m15_ICE1m15")
        assert matched_binding_window(tr, tc, lc8) == 1  # BSN2 in PEP1_BW

    def test_d_het_vs_single(self, lc8):
        tr = parse_tag("BSN2m15_SPAG5m15")
        tc = parse_tag("BSN2s15")
        mbw = matched_binding_window(tr, tc, lc8)
        assert mbw == 1  # BSN2 matches core1

    def test_matched_client(self, lc8):
        tr = parse_tag("BSN2m15_SPAG5m15")
        tc = parse_tag("BSN2s15")
        mbw = matched_binding_window(tr, tc, lc8)
        mc = matched_client_for_comparison(tr, tc, mbw, lc8)
        assert mc == "BSN2"
