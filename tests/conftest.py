"""
Shared test fixtures for dimer-ovl test suite.

Provides synthetic DimerSystem configs, in-memory H5 files, and OVL dicts
that don't depend on any real data files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pytest

from dimer_ovl.config import DimerSystem, lc8_system
from dimer_ovl.tags import Tag, parse_tag


# ── System fixtures ──────────────────────────────────────────────────


@pytest.fixture
def lc8() -> DimerSystem:
    """LC8/DYNLL1 system config."""
    return lc8_system()


@pytest.fixture
def tiny_system() -> DimerSystem:
    """Small 5-residue/chain dimer for fast unit tests."""
    return DimerSystem(
        chain_length=5,
        peptide_seqs={"PEP1": "ACDEF", "PEP2": "GHIKL"},
        binding_windows_1=[(0, 2), (7, 9)],
        binding_windows_2=[(3, 4), (5, 6)],
    )


@pytest.fixture
def medium_system() -> DimerSystem:
    """20-residue/chain dimer for integration-level tests."""
    return DimerSystem(
        chain_length=20,
        peptide_seqs={"AAA": "AAAAA", "BBB": "BBBBB"},
        binding_windows_1=[(0, 9), (30, 39)],
        binding_windows_2=[(10, 19), (20, 29)],
    )


# ── Tag fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tag_s() -> Tag:
    return parse_tag("BSN2s15")


@pytest.fixture
def tag_d() -> Tag:
    return parse_tag("BSN2d15")


@pytest.fixture
def tag_m() -> Tag:
    return parse_tag("BSN2m15_SPAG5m15")


# ── Synthetic OVL dict fixtures ──────────────────────────────────────


def _make_ovl_dict(
    n: int,
    n_rep_pairs: int = 3,
    tag: str = "BSN2d15",
    base_ovl: float = 0.5,
    noise: float = 0.05,
    seed: int = 42,
    include_dimer_keys: bool = True,
) -> Dict[str, np.ndarray]:
    """Create a synthetic OVL dict with realistic key structure.

    Returns view-pair keys like ``BSN2d15_r1_PLmat1_BSN2_vs_BSN2d15_r2_PLmat1_BSN2``
    and dimer keys like ``BSN2d15_r1_dimer_vs_BSN2d15_r2_dimer``.
    """
    rng = np.random.default_rng(seed)
    result: Dict[str, np.ndarray] = {}
    reps = list(range(1, n_rep_pairs + 2))  # enough reps for n_rep_pairs cross-pairs

    pair_idx = 0
    for i, ra in enumerate(reps):
        for rb in reps[i + 1:]:
            if pair_idx >= n_rep_pairs:
                break
            # Full view-pair matrix
            M = np.clip(
                base_ovl + rng.normal(0, noise, size=(n, n)),
                0.0, 1.0,
            ).astype(np.float32)
            # Make symmetric
            M = (M + M.T) / 2

            key = f"{tag}_r{ra}_PLmat1_BSN2_vs_{tag}_r{rb}_PLmat1_BSN2"
            result[key] = M

            if include_dimer_keys:
                dimer_n = min(n, 10)  # use first 10 as "dimer" block
                dk = f"{tag}_r{ra}_dimer_vs_{tag}_r{rb}_dimer"
                result[dk] = M[:dimer_n, :dimer_n].copy()

            pair_idx += 1
        if pair_idx >= n_rep_pairs:
            break

    return result


@pytest.fixture
def synthetic_ovl_20():
    """20×20 OVL dict with 3 rep-pairs."""
    return _make_ovl_dict(20, n_rep_pairs=3)


@pytest.fixture
def synthetic_ovl_pair():
    """Pair of OVL dicts (cross, intra) for KS testing."""
    cross = _make_ovl_dict(20, n_rep_pairs=9, base_ovl=0.4, seed=100)
    intra = _make_ovl_dict(20, n_rep_pairs=3, base_ovl=0.7, seed=200)
    return cross, intra


# ── Synthetic distance H5 fixtures ───────────────────────────────────


@pytest.fixture
def dist_h5_path(tmp_path) -> Path:
    """Create a synthetic distance H5 file with resids."""
    n_atoms = 25  # 10 dimer + 5 ligA + 5 ligB + 5 ligB(other view)
    n_frames = 50
    rng = np.random.default_rng(42)

    # Resids: dimer=1-10, ligA=11-15, ligB=16-20
    resids = np.arange(1, n_atoms + 1, dtype=np.int32)
    # Random distances
    distances = rng.normal(10.0, 2.0, size=(n_atoms, n_atoms, n_frames))
    distances = np.abs(distances)  # positive distances

    p = tmp_path / "test_dist.h5"
    with h5py.File(p, "w") as h5:
        h5.create_dataset("distances", data=distances.astype(np.float32))
        h5.create_dataset("resids", data=resids)
    return p


@pytest.fixture
def hist_h5_pair(tmp_path) -> tuple:
    """Create a pair of synthetic histogram H5 files with shared edges."""
    n = 15
    n_bins = 10
    rng = np.random.default_rng(42)

    edges = np.zeros((n, n, n_bins + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            edges[i, j, :] = np.linspace(0, 20, n_bins + 1)

    for label in ("A", "B"):
        counts = rng.poisson(50, size=(n, n, n_bins)).astype(np.int64)
        p = tmp_path / f"hist_{label}.h5"
        with h5py.File(p, "w") as h5:
            h5.create_dataset("hist_counts", data=counts)
            h5.create_dataset("bin_edges", data=edges)

    return tmp_path / "hist_A.h5", tmp_path / "hist_B.h5"
