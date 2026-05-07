"""
Pre- and post-checks for histogram generation.

Pre-check: verify resids array consistency before computation.
Post-check: spot-check histogram weighted means against distance matrix means.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np

from coop_stat.histogram.indexing import identify_ligand_atoms
from coop_stat.io.h5 import pick_3d_dataset, read_resids
from coop_stat.tags import Tag


def pre_check_resids(
    tag: Tag,
    rep: int,
    view: str,
    dist_h5_path: Path,
    expected_canon: int,
    resid_dset: str = "resids",
    dist_dset: Optional[str] = None,
) -> Dict:
    """Verify resids array before computation.

    Checks:
        (a) ``resids`` dataset exists
        (b) Length matches raw_n
        (c) For d/m tags: correct ligand atom count
        (d) Canonical size matches expected value
        (e) No duplicate resids within groups

    Returns result dict with ``status`` = ``"pass"`` or ``"FAIL"``.
    """
    label = f"{tag}_r{rep}_{view}"
    peplen = tag.peptide_length
    issues: List[str] = []

    import h5py
    with h5py.File(dist_h5_path, "r") as h5:
        dset_name = pick_3d_dataset(h5, preferred=dist_dset)
        raw_n = h5[dset_name].shape[0]
        has_resids = resid_dset in h5
        resids = h5[resid_dset][...].astype(int) if has_resids else None

    result: Dict = {
        "label": label,
        "dist_h5": str(dist_h5_path),
        "raw_n": raw_n,
        "has_resids": has_resids,
        "state": tag.state.value,
        "peplen": peplen,
        "expected_canon": expected_canon,
        "issues": issues,
        "status": None,
    }

    if not has_resids:
        issues.append(f"'{resid_dset}' dataset missing from {dist_h5_path}")
        result["status"] = "FAIL"
        return result

    if len(resids) != raw_n:
        issues.append(f"len(resids)={len(resids)} != raw_n={raw_n}")

    if tag.state.value != "s":
        dimer_raw, ligA_raw, ligB_raw = identify_ligand_atoms(resids, peplen)
        if len(ligA_raw) != peplen:
            issues.append(f"Expected {peplen} ligA atoms, got {len(ligA_raw)}")
        if len(ligB_raw) != peplen:
            issues.append(f"Expected {peplen} ligB atoms, got {len(ligB_raw)}")
        for grp_name, grp in [
            ("dimer", dimer_raw), ("ligA", ligA_raw), ("ligB", ligB_raw)
        ]:
            r = resids[grp]
            if len(set(r)) != len(r):
                issues.append(f"Duplicate resids in {grp_name} group")
        canon_n = len(dimer_raw) + peplen
    else:
        canon_n = raw_n

    if canon_n != expected_canon:
        issues.append(
            f"Derived canonical size {canon_n} != expected {expected_canon}"
        )

    result["derived_canon"] = canon_n
    result["status"] = "FAIL" if issues else "pass"
    return result


def post_check_histogram(
    hist_h5: Path,
    dist_h5: Path,
    canon_idx: np.ndarray,
    dimer_n: int,
    n_check: int = 20,
    tol: float = 1.5,
    counts_dset: str = "hist_counts",
    edges_dset: str = "bin_edges",
    dist_dset: Optional[str] = None,
    start_frame: int = 0,
    seed: int = 42,
) -> Dict:
    """Spot-check histogram weighted mean vs distance matrix mean.

    Checks DD, DL, and LL blocks separately.

    Returns result dict with ``status`` = ``"pass"`` or ``"FAIL"``.
    """
    rng = np.random.default_rng(seed)
    n_canon = len(canon_idx)
    has_lig = n_canon > dimer_n
    dimer_lim = min(dimer_n, n_canon)

    import h5py
    with h5py.File(dist_h5, "r") as h5:
        preferred = pick_3d_dataset(h5, preferred=dist_dset)

    # Generate test pairs for each block
    dd_pairs = _sample_pairs(rng, 0, dimer_lim, n_check)
    dl_pairs = _sample_cross_pairs(rng, 0, dimer_lim, dimer_lim, n_canon, n_check // 2) if has_lig else []
    ll_pairs = _sample_pairs(rng, dimer_lim, n_canon, n_check // 2) if has_lig and n_canon - dimer_lim >= 2 else []

    results: List[Dict] = []
    failures: List[Dict] = []

    for ci, cj, block in (
        [(i, j, "DD") for i, j in dd_pairs]
        + [(i, j, "DL") for i, j in dl_pairs]
        + [(i, j, "LL") for i, j in ll_pairs]
    ):
        hist_mean = _hist_weighted_mean(hist_h5, counts_dset, edges_dset, ci, cj)
        ri, rj = int(canon_idx[ci]), int(canon_idx[cj])
        dist_mean = _dist_pair_mean(dist_h5, preferred, ri, rj, start_frame)

        if hist_mean is None:
            results.append({
                "block": block, "ci": ci, "cj": cj,
                "status": "skipped_empty",
            })
            continue

        delta = abs(hist_mean - dist_mean)
        passed = delta <= tol
        entry = {
            "block": block, "ci": ci, "cj": cj, "ri": ri, "rj": rj,
            "hist_mean": round(hist_mean, 4),
            "dist_mean": round(dist_mean, 4),
            "delta_ang": round(delta, 4),
            "status": "pass" if passed else "FAIL",
        }
        results.append(entry)
        if not passed:
            failures.append(entry)

    return {
        "n_pass": sum(1 for r in results if r["status"] == "pass"),
        "n_fail": len(failures),
        "status": "FAIL" if failures else "pass",
        "pair_results": results,
        "failures": failures,
    }


# ── helpers ──────────────────────────────────────────────────────────

def _sample_pairs(rng, lo, hi, n):
    if hi - lo < 2:
        return []
    raw = rng.integers(lo, hi, size=(n * 4, 2))
    return [(int(r[0]), int(r[1])) for r in raw if r[0] < r[1]][:n]


def _sample_cross_pairs(rng, lo1, hi1, lo2, hi2, n):
    i_idx = rng.integers(lo1, hi1, size=n)
    j_idx = rng.integers(lo2, hi2, size=n)
    return [(int(i), int(j)) for i, j in zip(i_idx, j_idx)]


def _hist_weighted_mean(h5_path, counts_dset, edges_dset, i, j):
    import h5py
    with h5py.File(h5_path, "r") as h5:
        c = h5[counts_dset][i, j, :].astype(float)
        e = h5[edges_dset][i, j, :]
    s = c.sum()
    if s == 0:
        return None
    centers = 0.5 * (e[:-1] + e[1:])
    return float((c * centers).sum() / s)


def _dist_pair_mean(h5_path, dset_name, ri, rj, start_frame):
    import h5py
    with h5py.File(h5_path, "r") as h5:
        D = h5[dset_name]
        if D.shape[0] > D.shape[-1]:
            trace = D[start_frame:, ri, rj]
        else:
            trace = D[ri, rj, start_frame:]
    return float(np.asarray(trace, dtype=float).mean())
