"""
Histogram construction from distance matrices.

Two-pass workflow:
    1. ``compute_edges`` — scan all systems to find global min/max per pair,
       then build shared canonical bin edges.
    2. ``build_histogram`` — use the shared edges to histogram each (tag, rep, view).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


import numpy as np

from dimer_ovl.histogram.indexing import (
    build_resid_aware_index,
    canonical_size,
    get_canonical_index,
)
from dimer_ovl.io.h5 import pick_3d_dataset, read_resids
from dimer_ovl.tags import Tag


# ── Pass 1: edge computation ─────────────────────────────────────────


def compute_pairwise_minmax(
    tagged_paths: List[Tuple[Tag, str, Path]],
    n_canon: int,
    dist_dset: Optional[str] = None,
    start_frame: int = 0,
    resid_dset: str = "resids",
    include_diagonal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scan all (tag, view, dist_h5) to find global min/max for each pair.

    Parameters
    ----------
    tagged_paths : list of (Tag, view, dist_h5_path)
        All system/view/file combinations to scan.
    n_canon : int
        Expected canonical matrix size (must match across all systems).
    dist_dset : str, optional
        Preferred 3-D dataset name.
    start_frame : int
        First frame to include.
    resid_dset : str
        Dataset name for residue IDs.
    include_diagonal : bool
        Whether to include self-pairs.

    Returns
    -------
    dmin, dmax : ndarray of shape (n_canon, n_canon)
    """
    dmin = np.full((n_canon, n_canon), np.nan, dtype=np.float64)
    dmax = np.full((n_canon, n_canon), np.nan, dtype=np.float64)

    for tag, view, dist_path in tagged_paths:
        canon_idx, _, _ = get_canonical_index(
            tag, view, dist_path, resid_dset=resid_dset, dist_dset=dist_dset
        )
        if len(canon_idx) != n_canon:
            raise ValueError(
                f"Canonical size mismatch for {tag} {view}: "
                f"{len(canon_idx)} != {n_canon}"
            )

        import h5py
        with h5py.File(dist_path, "r") as h5:
            dset_name = pick_3d_dataset(h5, preferred=dist_dset)
            chunk = h5[dset_name][:, :, start_frame:]
            V = chunk[np.ix_(canon_idx, canon_idx, np.arange(chunk.shape[2]))]

        local_min = np.nanmin(V, axis=-1)
        local_max = np.nanmax(V, axis=-1)

        if not include_diagonal:
            ii = np.arange(n_canon)
            local_min[ii, ii] = np.nan
            local_max[ii, ii] = np.nan

        # Merge into global min/max
        mask_lo = np.isfinite(local_min)
        both = np.isfinite(dmin) & mask_lo
        dmin[both] = np.minimum(dmin[both], local_min[both])
        dmin[~np.isfinite(dmin) & mask_lo] = local_min[~np.isfinite(dmin) & mask_lo]

        mask_hi = np.isfinite(local_max)
        both = np.isfinite(dmax) & mask_hi
        dmax[both] = np.maximum(dmax[both], local_max[both])
        dmax[~np.isfinite(dmax) & mask_hi] = local_max[~np.isfinite(dmax) & mask_hi]

    return dmin, dmax


def build_edges(
    dmin: np.ndarray, dmax: np.ndarray, n_bins: int, pad: float = 1e-6,
) -> np.ndarray:
    """Build per-pair bin edges from global min/max.

    Returns shape ``(n, n, n_bins+1)`` float64.
    """
    n = dmin.shape[0]
    edges = np.empty((n, n, n_bins + 1), dtype=np.float64)
    lo = dmin - pad
    hi = dmax + pad
    bad = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi <= lo)
    lo[bad] = 0.0
    hi[bad] = 1e-3
    for i in range(n):
        for j in range(n):
            edges[i, j, :] = np.linspace(lo[i, j], hi[i, j], n_bins + 1)
    return edges


def write_edges_h5(
    path: Path,
    edges: np.ndarray,
    edges_dset: str = "bin_edges",
    n_bins: int = 60,
    start_frame: int = 0,
    tags: Optional[List[str]] = None,
) -> Path:
    """Write canonical bin edges to H5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(path, "w") as h5:
        h5.create_dataset(edges_dset, data=edges, compression="gzip", shuffle=True)
        h5.attrs["bins"] = n_bins
        h5.attrs["start_frame"] = start_frame
        h5.attrs["canonical_size"] = edges.shape[0]
        if tags:
            h5.attrs["tags"] = ",".join(tags)
    return path


# ── Pass 2: histogram construction ───────────────────────────────────


def build_histogram(
    tag: Tag,
    view: str,
    dist_h5: Path,
    canon_edges: np.ndarray,
    start_frame: int = 0,
    dist_dset: Optional[str] = None,
    resid_dset: str = "resids",
    include_diagonal: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Build per-pair distance histograms for one (tag, view).

    Parameters
    ----------
    tag : Tag
    view : str
    dist_h5 : Path
    canon_edges : ndarray of shape (n_canon, n_canon, n_bins+1)
    start_frame : int
    dist_dset : str, optional
    resid_dset : str
    include_diagonal : bool

    Returns
    -------
    counts : ndarray (n_canon, n_canon, n_bins) int64
    edges : ndarray (n_canon, n_canon, n_bins+1) float64
    canon_idx : ndarray (n_canon,) int64
    canon_resids : ndarray (n_canon,) int32 or None
    """
    canon_idx, canon_resids, _ = get_canonical_index(
        tag, view, dist_h5, resid_dset=resid_dset, dist_dset=dist_dset
    )
    n_canon = len(canon_idx)
    n_bins = canon_edges.shape[-1] - 1

    if canon_edges.shape[0] != n_canon:
        raise ValueError(
            f"Edge/index size mismatch: edges={canon_edges.shape[0]} "
            f"vs canon={n_canon}"
        )

    counts = np.zeros((n_canon, n_canon, n_bins), dtype=np.int64)
    out_edges = np.zeros((n_canon, n_canon, n_bins + 1), dtype=np.float64)

    import h5py
    with h5py.File(dist_h5, "r") as h5:
        dset_name = pick_3d_dataset(h5, preferred=dist_dset)
        D = h5[dset_name]

        for ci in range(n_canon):
            ri = int(canon_idx[ci])
            for cj in range(n_canon):
                rj = int(canon_idx[cj])
                e = canon_edges[ci, cj, :]
                out_edges[ci, cj, :] = e
                if ci == cj and not include_diagonal:
                    continue
                shape = D.shape
                if shape[0] > shape[-1]:
                    x = D[start_frame:, ri, rj]
                else:
                    x = D[ri, rj, start_frame:]
                c, _ = np.histogram(x, bins=e)
                counts[ci, cj, :] = c

    return counts, out_edges, canon_idx, canon_resids


def write_histogram_h5(
    path: Path,
    counts: np.ndarray,
    edges: np.ndarray,
    tag: Tag,
    rep: int,
    view: str,
    canon_resids: Optional[np.ndarray] = None,
    counts_dset: str = "hist_counts",
    edges_dset: str = "bin_edges",
) -> Path:
    """Write a histogram H5 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import h5py
    with h5py.File(path, "w") as h5:
        h5.create_dataset(counts_dset, data=counts, compression="gzip", shuffle=True)
        h5.create_dataset(edges_dset, data=edges, compression="gzip", shuffle=True)
        if canon_resids is not None:
            h5.create_dataset(
                "canonical_resids", data=canon_resids,
                compression="gzip", shuffle=True,
            )
        h5.attrs["tag"] = str(tag)
        h5.attrs["rep"] = rep
        h5.attrs["view"] = view
        h5.attrs["canonical_size"] = counts.shape[0]
        h5.attrs["resid_aware_index"] = canon_resids is not None
    return path
