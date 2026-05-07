"""
Core OVL (distributional overlap) computation.

OVL between two distance distributions is the integral of
``min(P_A(x), P_B(x))`` — computed from histogram bin counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def ovl_from_counts(countsA: np.ndarray, countsB: np.ndarray) -> np.ndarray:
    """Compute pairwise OVL from histogram counts.

    Parameters
    ----------
    countsA, countsB : ndarray of shape (n, n, n_bins)
        Bin counts for two histogram views.

    Returns
    -------
    ndarray of shape (n, n) float32
        OVL value for each residue pair.
    """
    if countsA.shape != countsB.shape:
        raise ValueError(f"Shape mismatch: {countsA.shape} vs {countsB.shape}")
    sumA = countsA.sum(axis=-1, keepdims=True)
    sumB = countsB.sum(axis=-1, keepdims=True)
    pA = np.divide(countsA, sumA, out=np.zeros_like(countsA, dtype=float), where=sumA > 0)
    pB = np.divide(countsB, sumB, out=np.zeros_like(countsB, dtype=float), where=sumB > 0)
    return np.minimum(pA, pB).sum(axis=-1).astype(np.float32)


def ovl_from_hist_h5(
    path_A: Path,
    path_B: Path,
    counts_dset: str = "hist_counts",
    edges_dset: str = "bin_edges",
    start: int = 0,
    stop: Optional[int] = None,
    edges_atol: float = 1e-7,
) -> np.ndarray:
    """Compute OVL matrix from two histogram H5 files.

    Parameters
    ----------
    path_A, path_B : Path
        Histogram H5 files.
    counts_dset, edges_dset : str
        Dataset names inside the H5 files.
    start, stop : int
        Slice range in canonical index space (default: full matrix).
    edges_atol : float
        Tolerance for bin edge comparison.

    Returns
    -------
    ndarray of shape (stop-start, stop-start) float32
    """
    cA, eA = _load_hist_view(path_A, counts_dset, edges_dset, start, stop)
    cB, eB = _load_hist_view(path_B, counts_dset, edges_dset, start, stop)

    if eA.shape != eB.shape or not np.allclose(eA, eB, rtol=0.0, atol=edges_atol):
        raise ValueError(
            f"Bin edges differ between {path_A} and {path_B}. "
            "Regenerate histograms with identical edges."
        )
    return ovl_from_counts(cA, cB)


def _load_hist_view(path, counts_dset, edges_dset, start, stop):
    """Load histogram counts and edges, sliced to [start:stop, start:stop]."""
    import h5py
    from coop_stat.io.h5 import pick_3d_dataset
    with h5py.File(path, "r") as h5:
        # Fall back to first 3-D dataset if named one is missing
        if counts_dset not in h5:
            counts_dset = pick_3d_dataset(h5, preferred=counts_dset)
        if edges_dset not in h5:
            raise KeyError(f"No edges dataset '{edges_dset}' in {path}")
        counts = h5[counts_dset][...].astype(np.float64)
        edges = h5[edges_dset][...].astype(np.float64)

    n = counts.shape[0]
    if stop is None:
        stop = n
    if not (0 <= start < stop <= n):
        raise ValueError(f"Slice {start}:{stop} invalid for N={n}")
    return counts[start:stop, start:stop, :], edges[start:stop, start:stop, :]
