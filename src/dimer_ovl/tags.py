"""
Tag parsing and metadata for simulation system identifiers.

A *tag* encodes system identity, binding state, and peptide length:
    BSN2s15     → singly-bound BSN2, 15-residue peptide
    BSN2d15     → doubly-bound homodimer BSN2
    BSN2m15_SPAG5m15 → mixed (heterodimeric ligand) BSN2 + SPAG5

Tags are the primary key linking histograms, OVL matrices, and comparisons.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class SymClass(str, Enum):
    """Symmetry classification for KS sample collection."""
    SINGLE = "s"           # singly bound — one peptide
    D_HOMO = "d_homo"      # doubly bound, same peptide both grooves
    D_HET = "d_het"        # doubly bound, different peptides (mixed tag)


class BindingState(str, Enum):
    SINGLY = "s"
    DOUBLY = "d"
    MIXED = "m"


@dataclass(frozen=True)
class Tag:
    """Parsed simulation system tag.

    Attributes
    ----------
    normalized : str
        Canonical string form (e.g. ``"BSN2s15"``).
    kind : str
        ``"single"`` or ``"mixed"``.
    state : BindingState
        Singly, doubly, or mixed bound.
    peptide_length : int
        Number of residues per peptide ligand.
    sym_class : SymClass
        C2 symmetry classification for KS sampling.
    core : str or None
        Client name for single tags (e.g. ``"BSN2"``).
    core1 : str or None
        First client for mixed tags.
    core2 : str or None
        Second client for mixed tags.
    """

    normalized: str
    kind: str
    state: BindingState
    peptide_length: int
    sym_class: SymClass
    core: Optional[str] = None
    core1: Optional[str] = None
    core2: Optional[str] = None

    @property
    def display_core(self) -> str:
        """Human-readable core name."""
        if self.kind == "single":
            return self.core or self.normalized
        return f"{self.core1}/{self.core2}"

    @property
    def is_mixed(self) -> bool:
        return self.kind == "mixed"

    @property
    def comparison_views(self) -> list[str]:
        """View tokens for OVL comparison (PLmat notation)."""
        if self.state == BindingState.SINGLY:
            return ["native"]
        return ["PLmat1", "PLmat2as1"]

    @property
    def histogram_views(self) -> list[str]:
        """View tokens as used in histogram filenames (on-disk notation)."""
        if self.state == BindingState.SINGLY:
            return ["native"]
        return ["site1", "site2as1"]

    def __str__(self) -> str:
        return self.normalized


# ── Regexes ──────────────────────────────────────────────────────────

_SINGLE_RE = re.compile(r"^([A-Za-z0-9]+)([sSdD])(\d+)$")
_MIXED_RE = re.compile(
    r"^([A-Za-z0-9]+)[mM](\d+)_([A-Za-z0-9]+)[mM](\d+)$"
)
_LC8_PREFIX = re.compile(r"^LC8_", re.IGNORECASE)


# ── Public API ───────────────────────────────────────────────────────

def parse_tag(raw: str) -> Tag:
    """Parse a raw tag string into a structured Tag.

    Accepts forms like ``BSN2s15``, ``BSN2d15``, ``BSN2m15_SPAG5m15``,
    with optional ``LC8_`` prefix.

    Parameters
    ----------
    raw : str
        Raw tag string.

    Returns
    -------
    Tag
        Parsed tag with normalized form and metadata.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a valid tag.
    """
    s = _LC8_PREFIX.sub("", raw.strip())

    m = _SINGLE_RE.match(s)
    if m:
        core = m.group(1).upper()
        state_char = m.group(2).lower()
        plen = int(m.group(3))
        state = BindingState.SINGLY if state_char == "s" else BindingState.DOUBLY
        sym = SymClass.SINGLE if state == BindingState.SINGLY else SymClass.D_HOMO
        normalized = f"{core}{state_char}{plen}"
        return Tag(
            normalized=normalized,
            kind="single",
            state=state,
            peptide_length=plen,
            sym_class=sym,
            core=core,
        )

    m = _MIXED_RE.match(s)
    if m:
        c1, l1 = m.group(1).upper(), int(m.group(2))
        c2, l2 = m.group(3).upper(), int(m.group(4))
        if l1 != l2:
            raise ValueError(
                f"Mixed tag requires equal peptide lengths: {raw} "
                f"(got {l1} vs {l2})"
            )
        normalized = f"{c1}m{l1}_{c2}m{l2}"
        return Tag(
            normalized=normalized,
            kind="mixed",
            state=BindingState.MIXED,
            peptide_length=l1,
            sym_class=SymClass.D_HET,
            core1=c1,
            core2=c2,
        )

    raise ValueError(f"Cannot parse tag '{raw}'")


def looks_like_tag(s: str) -> bool:
    """Return True if *s* looks like a parseable tag string."""
    s = _LC8_PREFIX.sub("", s.strip())
    return bool(_SINGLE_RE.match(s) or _MIXED_RE.match(s))


def make_tag(core: str, state: str, peptide_length: int) -> Tag:
    """Convenience constructor for single tags.

    >>> make_tag("BSN2", "s", 15)
    Tag(normalized='BSN2s15', ...)
    """
    return parse_tag(f"{core}{state}{peptide_length}")


def dd_sym_applies(tag_r: Tag, tag_c: Tag) -> bool:
    """True when C2 symmetric DD expansion is valid for this tag pair.

    Returns True when at least one tag is d_homo (OR logic).
    See v75 docstring NOTE on Fix 3 for rationale.
    """
    return (
        tag_r.sym_class == SymClass.D_HOMO
        or tag_c.sym_class == SymClass.D_HOMO
    )
