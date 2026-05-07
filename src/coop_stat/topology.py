"""
Topology operations for symmetric homodimers.

Handles C2 symmetry, pair equivalents, binding window occupancy,
and matched-client logic for cross-comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from coop_stat.config import DimerSystem
from coop_stat.io.gro import gro_sequence_for_range, parse_ligand_ranges
from coop_stat.tags import BindingState, SymClass, Tag


def pair_equivalents(
    i: int,
    j: int,
    system: DimerSystem,
    expand_sym: bool = False,
) -> List[Tuple[int, int, str]]:
    """Return equivalent (i, j, label) tuples under C2 symmetry.

    Parameters
    ----------
    i, j : int
        Residue indices (0-based).
    system : DimerSystem
        Dimer topology.
    expand_sym : bool
        If True, include the C2-partner pair when it differs from (i, j).

    Returns
    -------
    list of (i, j, label) where label is ``"direct"`` or ``"sym"``.
    """
    nres = system.dimer_n
    out = []
    seen = set()
    ii, jj = (min(i, j), max(i, j))
    if 0 <= ii < nres and 0 <= jj < nres:
        out.append((ii, jj, "direct"))
        seen.add((ii, jj))
    if expand_sym:
        mi = system.c2_mate(i)
        mj = system.c2_mate(j)
        if mi is not None and mj is not None:
            si, sj = min(mi, mj), max(mi, mj)
            if 0 <= si < nres and 0 <= sj < nres and (si, sj) not in seen:
                out.append((si, sj, "sym"))
    return out


def is_dimer_pair(i: int, j: int, system: DimerSystem) -> bool:
    """True if both indices are in the dimer block (< dimer_n)."""
    return i < system.dimer_n and j < system.dimer_n


# ── window occupancy ─────────────────────────────────────────────────


def tag_window_occupants(
    tag: Tag,
    system: DimerSystem,
    top_dir: Optional[str] = None,
    ligand_ranges: Optional[str] = None,
) -> Dict[int, Optional[str]]:
    """Determine which client occupies each binding window for a tag.

    Returns {1: client_or_None, 2: client_or_None}.
    """
    if tag.sym_class == SymClass.D_HOMO:
        return {1: tag.core, 2: tag.core}
    if tag.sym_class == SymClass.D_HET:
        return {system.pep1_bw: tag.core1, system.pep2_bw: tag.core2}
    # Singly bound: try to read from topology
    if top_dir and ligand_ranges:
        try:
            gro = _find_gro(tag, top_dir)
            ranges = parse_ligand_ranges(ligand_ranges)
            for (ra, rb), pbw in zip(ranges, [system.pep1_bw, system.pep2_bw]):
                try:
                    seq = gro_sequence_for_range(gro, ra, rb, system.aa3_to_1)
                    client = system.identify_client(seq)
                    other_bw = 1 if pbw == 2 else 2
                    return {pbw: client, other_bw: None}
                except ValueError:
                    continue
        except FileNotFoundError:
            pass
    return {1: None, 2: None}


def matched_binding_window(tag_r: Tag, tag_c: Tag, system: DimerSystem) -> int:
    """Determine which binding window to compare for a tag pair.

    Returns 0 (both), 1 (BW1), or 2 (BW2).
    """
    sr, sc = tag_r.sym_class, tag_c.sym_class
    if sr == SymClass.D_HET and sc == SymClass.D_HET:
        if tag_r.core1 == tag_c.core1:
            return system.pep1_bw
        if tag_r.core2 == tag_c.core2:
            return system.pep2_bw
        return 0
    # Find the d_het tag (if any) and the other
    if sr == SymClass.D_HET:
        mi, oi = tag_r, tag_c
    elif sc == SymClass.D_HET:
        mi, oi = tag_c, tag_r
    else:
        return 0
    oc = oi.core or ""
    if mi.core1 == oc:
        return system.pep1_bw
    if mi.core2 == oc:
        return system.pep2_bw
    return 0


def matched_client_for_comparison(
    tag_r: Tag, tag_c: Tag, mbw: int, system: DimerSystem,
) -> Optional[str]:
    """Return the client name that should be used for filtering DL/LL values."""
    if mbw == 0:
        return None
    for tag in [tag_r, tag_c]:
        if tag.sym_class == SymClass.D_HET:
            return tag.core1 if mbw == system.pep1_bw else tag.core2
    return tag_r.core


def window_roles(
    tag_r: Tag,
    tag_c: Tag,
    mbw: int,
    system: DimerSystem,
    top_dir: Optional[str] = None,
    ligand_ranges: Optional[str] = None,
) -> Dict[int, Dict]:
    """Describe the role of each binding window in a comparison."""
    ro = tag_window_occupants(tag_r, system, top_dir, ligand_ranges)
    co = tag_window_occupants(tag_c, system, top_dir, ligand_ranges)
    result = {}
    for bw in (1, 2):
        r, c = ro[bw], co[bw]
        inc = (mbw == 0) or (mbw == bw)
        if r is None and c is None:
            role = "unbound_in_both"
        elif r is None:
            role = "newly_engaged" if inc else "unbound_in_ref_excluded"
        elif c is None:
            role = "released" if inc else "bound_in_ref_excluded"
        elif r == c:
            role = "conserved_bound" if inc else "conserved_excluded"
        else:
            role = "different_clients_compared" if inc else "client_mismatch_excluded"
        result[bw] = {
            "ref_occupant": r,
            "cmp_occupant": c,
            "included": inc,
            "role": role,
        }
    return result


# ── helpers ──────────────────────────────────────────────────────────

def _find_gro(tag: Tag, top_dir: str) -> Path:
    """Locate the .gro topology file for a tag."""
    td = Path(top_dir)
    candidates = [
        td / f"LC8_{tag}_dros89_input_AC_calphas_adjusted.gro",
        td / f"LC8_{tag}_dros89_input_calphas_adjusted.gro",
        # Generic pattern for non-LC8 systems
        td / f"{tag}_calphas.gro",
        td / f"{tag}.gro",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No .gro topology found for tag '{tag}' in {top_dir}. "
        f"Checked: {[str(p) for p in candidates]}"
    )
