"""
Central configuration for a symmetric homodimer system.

Every pipeline stage receives a DimerSystem instead of hardcoded constants.
Users instantiate one for their protein and pass it through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DimerSystem:
    """Immutable description of a symmetric homodimer system.

    Parameters
    ----------
    chain_length : int
        Residue count per monomer chain (e.g. 89 for LC8/DYNLL1).
        dimer_n = 2 * chain_length.
    peptide_seqs : dict[str, str]
        Known peptide client sequences keyed by short name.
        Used to auto-identify clients from .gro topology.
    binding_windows_1 : list of (lo, hi) tuples
        0-based inclusive residue ranges comprising binding window 1
        (bound groove in singly-bound state).
    binding_windows_2 : list of (lo, hi) tuples
        0-based inclusive residue ranges comprising binding window 2
        (remote groove in singly-bound state).
    aa3_to_1 : dict[str, str]
        Three-letter → one-letter amino acid mapping.
        Defaults cover standard residues + CHARMM histidine variants.
    pep1_bw : int
        Which binding window the first (lower-resid) peptide occupies.
        Default 1.
    pep2_bw : int
        Which binding window the second (higher-resid) peptide occupies.
        Default 2.
    """

    chain_length: int
    peptide_seqs: Dict[str, str] = field(default_factory=dict)
    binding_windows_1: List[Tuple[int, int]] = field(default_factory=list)
    binding_windows_2: List[Tuple[int, int]] = field(default_factory=list)
    aa3_to_1: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_AA3))
    pep1_bw: int = 1
    pep2_bw: int = 2

    # ── derived properties ────────────────────────────────────────────

    @property
    def dimer_n(self) -> int:
        """Total residue count for the dimer (2 × chain_length)."""
        return 2 * self.chain_length

    @property
    def chain_a_range(self) -> Tuple[int, int]:
        """Half-open range [start, stop) for chain A indices."""
        return (0, self.chain_length)

    @property
    def chain_b_range(self) -> Tuple[int, int]:
        """Half-open range [start, stop) for chain B indices."""
        return (self.chain_length, self.dimer_n)

    # ── C2 symmetry ──────────────────────────────────────────────────

    def c2_mate(self, idx: int) -> Optional[int]:
        """Return the C2-symmetry partner index, or None if out of range."""
        if not (0 <= idx < self.dimer_n):
            return None
        mate = idx + self.chain_length if idx < self.chain_length else idx - self.chain_length
        return mate if 0 <= mate < self.dimer_n else None

    def c2_sym_map(self) -> np.ndarray:
        """1-D int32 array where ``sym_map[i]`` = C2 partner of dimer index *i*."""
        half = self.chain_length
        return np.concatenate([
            np.arange(half, self.dimer_n),
            np.arange(0, half),
        ]).astype(np.int32)

    # ── binding windows ──────────────────────────────────────────────

    def binding_indices(self, bw: int = 0) -> List[int]:
        """Return sorted residue indices for a binding window (0 = both)."""
        if bw == 1:
            windows = self.binding_windows_1
        elif bw == 2:
            windows = self.binding_windows_2
        elif bw == 0:
            windows = self.binding_windows_1 + self.binding_windows_2
        else:
            raise ValueError(f"bw must be 0, 1, or 2; got {bw}")
        idx = []
        for lo, hi in windows:
            idx.extend(range(lo, hi + 1))
        return sorted(idx)

    def residue_binding_window(self, idx: int) -> int:
        """Return which binding window (1, 2, or 0=neither) residue *idx* belongs to."""
        for lo, hi in self.binding_windows_1:
            if lo <= idx <= hi:
                return 1
        for lo, hi in self.binding_windows_2:
            if lo <= idx <= hi:
                return 2
        return 0

    # ── client identification ────────────────────────────────────────

    def identify_client(self, seq: str) -> str:
        """Match a peptide sequence to a known client name.

        Raises ValueError if zero or multiple matches found.
        """
        matches = [name for name, ref in self.peptide_seqs.items() if seq == ref]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Sequence matched multiple clients: {seq} → {matches}")
        raise ValueError(
            f"Cannot match sequence '{seq}' to known clients: "
            f"{list(self.peptide_seqs.keys())}"
        )

    # ── validation ───────────────────────────────────────────────────

    def __post_init__(self):
        if self.chain_length < 1:
            raise ValueError(f"chain_length must be positive, got {self.chain_length}")
        # Basic sanity: ranges must be non-negative and lo <= hi
        for label, windows in [("BW1", self.binding_windows_1),
                                ("BW2", self.binding_windows_2)]:
            for lo, hi in windows:
                if not (0 <= lo <= hi):
                    raise ValueError(
                        f"{label} range ({lo}, {hi}) invalid (need 0 <= lo <= hi)"
                    )


# ── LC8 preset ───────────────────────────────────────────────────────

def lc8_system() -> DimerSystem:
    """Pre-configured DimerSystem for Drosophila LC8/DYNLL1 (89 residues/chain)."""
    return DimerSystem(
        chain_length=89,
        peptide_seqs={
            "BSN2":  "YPRATAEFSTQTPSP",
            "BIM":   "YAMPSCDKSTQTPSP",
            "SPAG5": "YHPETQDSSTQTDTS",
            "ICE1":  "YEKELRHIGTQISSD",
        },
        binding_windows_1=[(1, 24), (59, 89), (114, 147)],
        binding_windows_2=[(25, 58), (90, 113), (148, 178)],
    )


# ── default amino acid map ───────────────────────────────────────────

_DEFAULT_AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H", "HSP": "H",
}
