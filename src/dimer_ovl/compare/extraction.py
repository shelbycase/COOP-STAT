"""
Extract OVL sample values for a residue pair from dictionaries of OVL matrices.

Implements the 5 KS sample collection rules:
    Rule 1: DD uses dimer keys (deduplicated by rep-pair); DL/LL uses view keys.
    Rule 2: Same-rep intra-tag excluded.
    Rule 3: d_het intra filtered by matched_client.
    Rule 4: DD sym expansion gated by dd_sym_applies (OR logic).
    Rule 5: Per-bucket means loaded where available.

Rule Descriptions:
1) What is deduplication by rep-pair? 
	We use the PLmat1 (protein-ligand matrix 1) and PLmat2as1 (protein-ligand matrix 2 as 1), 
	to preserve identical bin edges for the DL (dimer-to-ligand pairs) & LL (ligand1-to-ligand2) indices. 
	For an within-sepecies comparison (for sampling assessments), the DD (dimer-to-dimer pairs) are identical, 
	adding no new information. This is handled so that there is no trivial intra comparisons fed into 
	subsequent analyses. 
	
	example:
	BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2    ← view-pair 1
	BSN2d15_r1_PLmat2as1_BSN2_vs_BSN2d15_r2_PLmat2as1_BSN2  ← view-pair 2

2) What is the DD sym expansion? 
	the DD symmetry expansion allows for homotypic doubly bound (d_hom.) states to double the sample count for 
	KS testing. Since each monomer is identically bound, the same physical observable is being sampled. In this 
	case, the binding windows (including residues from opposite monomers) are accounted for, by adding i+89. 
	This treatment is triggered using an OR operator, such that if a single system is d_hom. the symmetry 
	expansion is applied. This doesn't apply to singly bound or d_het. states. 

3) What are per-bucket means?
	These are not used in the KS test, but the means are used in the filtering needed for statistical significance 
	(stat-sign) counts. The per-bucket means describe the asymmetric bound monomer environments for the d_het and 
	singly bound states. This is a correction that prevents erroneous symmetric averaging. 

	example (LC8_BSN2m15_SPAG5m15):
	The OVL calculator writes separate per-bucket means:
	ovl_mean_deduped_PLmat1_BSN2        ← only BSN2 view-pairs (esp. for DL&LL blocks), DD deduped (same dimeric values)
	ovl_mean_deduped_PLmat2as1_SPAG5    ← only SPAG5 view-pairs (esp. for DL&LL blocks), DD deduped
	ovl_mean_eligible_unblocked         ← everything averaged (diagnostic only)
	
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from dimer_ovl.config import DimerSystem


# ── OVL key classification ───────────────────────────────────────────


def is_dimer_key(key: str) -> bool:
    """True if *key* is a canonical dimer-block-only key."""
    return "_dimer_" in key or key.endswith("_dimer")


_TAG_PAT = r"(?:[A-Za-z0-9]+[sSdD]\d+|[A-Za-z0-9]+m\d+_[A-Za-z0-9]+m\d+)"
_OBJ_PAT = r"(r\d+(?:_[A-Za-z0-9]+)+)"
_PNR = re.compile(
    rf"^(?P<Lt>{_TAG_PAT})_(?P<Lo>{_OBJ_PAT})_vs_"
    rf"(?P<Rt>{_TAG_PAT})_(?P<Ro>{_OBJ_PAT})$",
    re.I,
)
_POR = re.compile(
    r"^(?P<Lt>.+?)r(?P<Lr>\d+)_vs_(?P<Rt>.+?)r(?P<Rr>\d+)$"
)


def parse_pair_key(key: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse an OVL H5 key into (left_tag, left_obj, right_tag, right_obj)."""
    m = _PNR.match(key)
    if m:
        return m.group("Lt"), m.group("Lo"), m.group("Rt"), m.group("Ro")
    m = _POR.match(key)
    if m:
        return (
            m.group("Lt"), f"r{int(m.group('Lr'))}",
            m.group("Rt"), f"r{int(m.group('Rr'))}",
        )
    return None


def rep_pair_from_key(key: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract (rep_A, rep_B) integers from an OVL key."""
    halves = key.split("_vs_", 1)
    if len(halves) != 2:
        return (None, None)
    return _extract_rep(halves[0]), _extract_rep(halves[1])


def _extract_rep(half: str) -> Optional[int]:
    m = re.search(r"(?:^|_)(r\d+)(?:_|$)", half)
    return int(m.group(1)[1:]) if m else None


def is_intra_same_rep(key: str) -> bool:
    """True if key represents same-tag, same-replicate comparison."""
    p = parse_pair_key(key)
    if p is None:
        return False
    Lt, _, Rt, _ = p
    if Lt != Rt:
        return False
    ra, rb = rep_pair_from_key(key)
    return ra is not None and rb is not None and ra == rb


# ── Client matching ──────────────────────────────────────────────────

_CLIENT_RE = re.compile(r"^r\d+_(?:PLmat1|PLmat2as1|native)_([A-Za-z0-9]+)$")


def client_from_view_obj(obj: str) -> Optional[str]:
    """Extract client name from a view-object string like ``r1_PLmat1_BSN2``."""
    m = _CLIENT_RE.match(obj)
    return m.group(1) if m else None


def client_match(key: str, matched_client: Optional[str]) -> bool:
    """True if key passes client filtering for *matched_client*."""
    if matched_client is None:
        return True
    if is_dimer_key(key):
        return True
    p = parse_pair_key(key)
    if p is None:
        return True
    _, Lo, _, Ro = p
    Lc = client_from_view_obj(Lo)
    Rc = client_from_view_obj(Ro)
    if Lc is None and Rc is None:
        return True
    if Lc is None:
        return Rc == matched_client
    if Rc is None:
        return Lc == matched_client
    return Lc == matched_client and Rc == matched_client


# ── Value extraction ─────────────────────────────────────────────────


def extract_values_for_pair(
    ovl_dict: Optional[Dict[str, np.ndarray]],
    i: int,
    j: int,
    system: DimerSystem,
    nres: Optional[int] = None,
    dd_sym: bool = False,
    matched_client: Optional[str] = None,
) -> List[float]:
    """Extract all OVL values for residue pair (i, j) from an OVL dict.

    Implements DD dedup (Rule 1), same-rep exclusion (Rule 2),
    client filtering (Rule 3), and sym expansion (Rule 4).

    Parameters
    ----------
    ovl_dict : dict of {key: ndarray}
        OVL matrices keyed by comparison label.
    i, j : int
        Residue pair indices.
    system : DimerSystem
        Topology config.
    nres : int, optional
        Matrix size (inferred from first matrix if None).
    dd_sym : bool
        Whether to include C2-symmetric pair values.
    matched_client : str, optional
        Client name for DL/LL filtering.

    Returns
    -------
    list of float
        All eligible OVL values for this pair.
    """
    if ovl_dict is None:
        return []

    dimer_n = system.dimer_n
    if nres is None:
        sample = next(
            (M for k, M in ovl_dict.items() if not is_dimer_key(k)),
            next(iter(ovl_dict.values()), None),
        )
        nres = sample.shape[0] if sample is not None else max(i, j) + 1

    vals: List[float] = []
    is_dd = i < dimer_n and j < dimer_n

    if is_dd:
        # Compute C2-partner pair
        mi, mj = system.c2_mate(i), system.c2_mate(j)
        if mi is not None and mj is not None:
            si, sj = min(mi, mj), max(mi, mj)
            sym_same = (si == min(i, j) and sj == max(i, j))
        else:
            si, sj, sym_same = None, None, True

        # Use dimer keys if available, else view keys
        dks = sorted(k for k in ovl_dict if is_dimer_key(k))
        vks = sorted(
            k for k in ovl_dict
            if not is_dimer_key(k) and client_match(k, matched_client)
        )
        src = dks if dks else vks
        seen: set = set()

        for key in src:
            M = ovl_dict[key]
            rp = rep_pair_from_key(key)
            if is_intra_same_rep(key) or rp in seen:
                continue
            seen.add(rp)
            n = M.shape[0]
            ii, jj = min(i, j), max(i, j)
            if ii < n and jj < n:
                vals.append(float(M[ii, jj]))
            if dd_sym and not sym_same and si is not None and si < n and sj < n:
                vals.append(float(M[si, sj]))
    else:
        for key, M in ovl_dict.items():
            if is_dimer_key(key) or is_intra_same_rep(key):
                continue
            if not client_match(key, matched_client):
                continue
            ii, jj = min(i, j), max(i, j)
            if ii < M.shape[0] and jj < M.shape[1]:
                vals.append(float(M[ii, jj]))

    return vals

def extract_labeled_values_for_pair(
    ovl_dict: Optional[Dict[str, np.ndarray]],
    i: int,
    j: int,
    system: DimerSystem,
    nres: Optional[int] = None,
    dd_sym: bool = False,
    matched_client: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Extract (label, value) pairs for per-key TSV columns.

    Same extraction logic as extract_values_for_pair, but returns
    the OVL key alongside each value. Same-rep entries get a
    ``|diagnostic`` suffix; sym-expanded entries get ``|sym``.
    """
    if ovl_dict is None:
        return []

    dimer_n = system.dimer_n
    if nres is None:
        sample = next(
            (M for k, M in ovl_dict.items() if not is_dimer_key(k)),
            next(iter(ovl_dict.values()), None),
        )
        nres = sample.shape[0] if sample is not None else max(i, j) + 1

    out: List[Tuple[str, float]] = []
    is_dd = i < dimer_n and j < dimer_n

    if is_dd:
        mi, mj = system.c2_mate(i), system.c2_mate(j)
        if mi is not None and mj is not None:
            si, sj = min(mi, mj), max(mi, mj)
            sym_same = (si == min(i, j) and sj == max(i, j))
        else:
            si, sj, sym_same = None, None, True

        vks = sorted(
            k for k in ovl_dict
            if not is_dimer_key(k) and client_match(k, matched_client)
        )
        dks = sorted(k for k in ovl_dict if is_dimer_key(k))

        # Group view keys by rep-pair
        r2v: Dict = {}
        for vk in vks:
            r2v.setdefault(rep_pair_from_key(vk), []).append(vk)

        # Map rep-pair to dimer-key matrix
        r2d: Dict = {}
        for dk in dks:
            r2d.setdefault(rep_pair_from_key(dk), ovl_dict[dk])

        seen: set = set()
        for rp in sorted(r2v):
            if rp in seen:
                continue
            seen.add(rp)
            vkeys = r2v[rp]
            fk = vkeys[0]
            sr = is_intra_same_rep(fk)
            Mv = r2d.get(rp, ovl_dict[fk])
            n = Mv.shape[0]
            ii, jj = min(i, j), max(i, j)

            vij = float(Mv[ii, jj]) if ii < n and jj < n else None
            vss = (
                float(Mv[si, sj])
                if dd_sym and not sym_same and si is not None and si < n and sj < n
                else None
            )

            if sr:
                if vij is not None:
                    for vk in vkeys:
                        out.append((f"{vk}|diagnostic", vij))
            else:
                if vij is not None:
                    for vk in vkeys:
                        out.append((vk, vij))
                if vss is not None:
                    out.append((f"{fk}|sym", vss))
    else:
        for key, M in ovl_dict.items():
            if is_dimer_key(key) or not client_match(key, matched_client):
                continue
            sr = is_intra_same_rep(key)
            ii, jj = min(i, j), max(i, j)
            if ii < M.shape[0] and jj < M.shape[1]:
                label = f"{key}|diagnostic" if sr else key
                out.append((label, float(M[ii, jj])))

    return out

def smart_mean(
    ovl_dict: Optional[Dict[str, np.ndarray]],
    matched_client: Optional[str] = None,
    exclude_same_rep_intra: bool = True,
) -> Optional[np.ndarray]:
    """Compute a naive mean over eligible view-pair matrices.

    Excludes dimer-only keys and (optionally) same-rep intra keys.
    Used as fallback when per-bucket means aren't available.
    """
    if not ovl_dict:
        return None
    mats = []
    for k, M in ovl_dict.items():
        if is_dimer_key(k):
            continue
        if exclude_same_rep_intra and is_intra_same_rep(k):
            continue
        if matched_client is not None and not client_match(k, matched_client):
            continue
        mats.append(M)
    return sum(mats) / len(mats) if mats else None
