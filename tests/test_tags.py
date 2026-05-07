"""Tests for coop_stat.tags — tag parsing, normalization, metadata."""

import pytest

from coop_stat.tags import (
    BindingState,
    SymClass,
    Tag,
    dd_sym_applies,
    looks_like_tag,
    make_tag,
    parse_tag,
)


class TestParseTag:
    """Tag parsing from raw strings."""

    def test_singly_bound(self):
        t = parse_tag("BSN2s15")
        assert t.normalized == "BSN2s15"
        assert t.kind == "single"
        assert t.state == BindingState.SINGLY
        assert t.peptide_length == 15
        assert t.sym_class == SymClass.SINGLE
        assert t.core == "BSN2"
        assert t.display_core == "BSN2"

    def test_doubly_bound(self):
        t = parse_tag("SPAG5d15")
        assert t.normalized == "SPAG5d15"
        assert t.state == BindingState.DOUBLY
        assert t.sym_class == SymClass.D_HOMO
        assert t.core == "SPAG5"

    def test_mixed(self):
        t = parse_tag("BSN2m15_SPAG5m15")
        assert t.normalized == "BSN2m15_SPAG5m15"
        assert t.kind == "mixed"
        assert t.state == BindingState.MIXED
        assert t.sym_class == SymClass.D_HET
        assert t.core1 == "BSN2"
        assert t.core2 == "SPAG5"
        assert t.display_core == "BSN2/SPAG5"

    def test_lc8_prefix_stripped(self):
        t = parse_tag("LC8_BSN2s15")
        assert t.normalized == "BSN2s15"

    def test_case_insensitive_state(self):
        t = parse_tag("BSN2S15")
        assert t.state == BindingState.SINGLY

    def test_mixed_unequal_lengths_raises(self):
        with pytest.raises(ValueError, match="equal peptide lengths"):
            parse_tag("BSN2m15_SPAG5m10")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_tag("not_a_tag")


class TestTagViews:
    """View token generation."""

    def test_singly_comparison_views(self):
        t = parse_tag("BSN2s15")
        assert t.comparison_views == ["native"]

    def test_doubly_comparison_views(self):
        t = parse_tag("BSN2d15")
        assert t.comparison_views == ["PLmat1", "PLmat2as1"]

    def test_singly_histogram_views(self):
        t = parse_tag("BSN2s15")
        assert t.histogram_views == ["native"]

    def test_doubly_histogram_views(self):
        t = parse_tag("BSN2d15")
        assert t.histogram_views == ["site1", "site2as1"]


class TestDdSymApplies:
    """C2 symmetry expansion gating — OR logic."""

    def test_s_vs_s_false(self):
        assert not dd_sym_applies(parse_tag("BSN2s15"), parse_tag("ICE1s15"))

    def test_d_vs_s_true(self):
        assert dd_sym_applies(parse_tag("BSN2d15"), parse_tag("BSN2s15"))

    def test_s_vs_d_true(self):
        assert dd_sym_applies(parse_tag("BSN2s15"), parse_tag("BSN2d15"))

    def test_d_vs_d_true(self):
        assert dd_sym_applies(parse_tag("BSN2d15"), parse_tag("SPAG5d15"))

    def test_d_het_vs_s_false(self):
        """d_het:s → both False (d_het is not d_homo)."""
        assert not dd_sym_applies(parse_tag("BSN2m15_SPAG5m15"), parse_tag("BSN2s15"))

    def test_d_het_vs_d_homo_true(self):
        assert dd_sym_applies(parse_tag("BSN2m15_SPAG5m15"), parse_tag("BSN2d15"))


class TestLooksLikeTag:
    def test_valid(self):
        assert looks_like_tag("BSN2s15")
        assert looks_like_tag("LC8_BSN2d15")
        assert looks_like_tag("BSN2m15_SPAG5m15")

    def test_invalid(self):
        assert not looks_like_tag("hello")
        assert not looks_like_tag("123")


class TestMakeTag:
    def test_simple(self):
        t = make_tag("BSN2", "s", 15)
        assert t.normalized == "BSN2s15"
        assert t.state == BindingState.SINGLY
