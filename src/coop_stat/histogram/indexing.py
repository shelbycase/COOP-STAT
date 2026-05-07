"""
Canonical index construction for mapping distance matrix indices to histogram
row/col positions.

For a symmetric homodimer with two bound peptides, the distance matrix has
``dimer_n + 2 * peplen`` atoms but each histogram view only includes
``dimer_n + peplen`` (dimer + one peptide).  The canonical index maps raw
distance matrix positions to view positions.

The resid-aware method auto-detects dimer/ligand atoms from residue IDs
in the distance H5, requiring no user-supplied ligand ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from coop_stat.tags import Tag


# ── public API ───────────────────────────────────────────────────────


def get_canonical_index(
    tag: Tag,
    view: str,
    dist_h5_path: Path,
    resid_dset: str = "resids",
    dist_dset: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """Return ``(canon_idx, canon_resids, resid_aware)`` for a (tag, view).

    Parameters
    ----------
    tag : Tag
        Parsed system tag.
    view : str
        Histogram view (``"native"``, ``"site1"``, ``"site2as1"``).
    dist_h5_path : Path
        Distance H5 file.
    resid_dset : str
        Dataset name for residue IDs in the dist H5.
    dist_dset : str, optional
        Preferred 3-D dataset name in dist H5.

    Returns
    -------
    canon_idx : ndarray of int64
        Maps canonical slot i → raw distance matrix index.
    canon_resids : ndarray of int32 or None
        Residue ID for each canonical slot (None if resids unavailable).
    resid_aware : bool
        True if resid-aware method was used.
    """
    import h5py
    from coop_stat.io.h5 import pick_3d_dataset, read_resids
    dist_resids = read_resids(dist_h5_path, resid_dset)

    with h5py.File(dist_h5_path, "r") as h5:
        raw_n = h5[pick_3d_dataset(h5, preferred=dist_dset)].shape[0]

    if dist_resids is not None:
        return build_resid_aware_index(tag, view, raw_n, dist_resids)

    # Arithmetic fallback
    import warnings
    warnings.warn(
        f"No '{resid_dset}' dataset in {dist_h5_path}; "
        f"using arithmetic fallback for {tag} {view}.",
        RuntimeWarning,
    )
    idx = build_arithmetic_index(tag, view, raw_n)
    return idx, None, False


def canonical_size(tag: Tag, raw_n: int) -> int:
    """Expected canonical matrix size for a tag given raw distance matrix size."""
    if tag.state.value == "s":
        return raw_n
    return raw_n - tag.peptide_length


# ── resid-aware index ────────────────────────────────────────────────


def identify_ligand_atoms(
    resids: np.ndarray, peplen: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Auto-identify dimer, ligandA, and ligandB raw atom indices from resids.

    The two peptide ligands have the highest residue IDs.  The top *peplen*
    resids are ligand B (higher-resid peptide); the next *peplen* are
    ligand A.  Each group is sorted ascending by resid.

    Returns (dimer_raw, ligA_raw, ligB_raw) as int64 arrays.
    """
    order_desc = np.argsort(resids)[::-1]
    ligand_all = order_desc[: 2 * peplen]
    ligB_unsorted = ligand_all[:peplen]
    ligA_unsorted = ligand_all[peplen : 2 * peplen]

    ligand_set = set(ligand_all.tolist())
    dimer_unsorted = np.array(
        [i for i in range(len(resids)) if i not in ligand_set], dtype=np.int64
    )

    dimer_raw = dimer_unsorted[np.argsort(resids[dimer_unsorted])]
    ligA_raw = ligA_unsorted[np.argsort(resids[ligA_unsorted])]
    ligB_raw = ligB_unsorted[np.argsort(resids[ligB_unsorted])]

    return dimer_raw.astype(np.int64), ligA_raw.astype(np.int64), ligB_raw.astype(np.int64)


def build_resid_aware_index(
    tag: Tag,
    view: str,
    raw_n: int,
    dist_resids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Resid-aware canonical index.

    For singly-bound: sorts all atoms by resid ascending.
    For doubly-bound/mixed: concatenates [dimer | ligandA_or_B].
    """
    if tag.state.value == "s":
        order = np.argsort(dist_resids).astype(np.int64)
        return order, dist_resids[order].astype(np.int32), True

    peplen = tag.peptide_length
    dimer_raw, ligA_raw, ligB_raw = identify_ligand_atoms(dist_resids, peplen)

    if view in ("site1", "PLmat1"):
        canon_idx = np.concatenate([dimer_raw, ligA_raw]).astype(np.int64)
    elif view in ("site2as1", "PLmat2as1"):
        canon_idx = np.concatenate([dimer_raw, ligB_raw]).astype(np.int64)
    else:
        raise ValueError(f"Unknown view '{view}' for doubly-bound tag {tag}")

    canon_resids = dist_resids[canon_idx].astype(np.int32)
    return canon_idx, canon_resids, True


def build_arithmetic_index(tag: Tag, view: str, raw_n: int) -> np.ndarray:
    """Arithmetic fallback (assumes standard atom ordering in dist H5).

    Use only when resids are unavailable.
    """
    if tag.state.value == "s":
        return np.arange(raw_n, dtype=np.int64)

    peplen = tag.peptide_length
    canon_n = canonical_size(tag, raw_n)
    protein_prefix = canon_n - peplen

    if view in ("site1", "PLmat1"):
        return np.arange(canon_n, dtype=np.int64)

    if view in ("site2as1", "PLmat2as1"):
        idx = np.empty(canon_n, dtype=np.int64)
        idx[:protein_prefix] = np.arange(protein_prefix, dtype=np.int64)
        idx[protein_prefix:] = np.arange(canon_n, raw_n, dtype=np.int64)
        return idx

    raise ValueError(f"Unknown view '{view}' for tag {tag}")
