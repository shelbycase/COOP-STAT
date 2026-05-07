"""
Block-aware mean OVL computation with per-(view, client) deduplication.

The OVL matrix has four blocks:
    DD (dimer×dimer)   — identical across view-pairs for a given rep-pair
    DL (dimer×ligand)  — view-dependent
    LD (ligand×dimer)  — derived as DL^T
    LL (ligand×ligand) — view-dependent

Per-bucket means avoid double-counting DD while preserving per-view DL/LL.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dimer_ovl.config import DimerSystem


def compute_block_aware_mean(
    records: List[Dict[str, Any]],
    system: DimerSystem,
    matrix_n: int,
    dd_atol: float = 1e-6,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], np.ndarray, Dict]:
    """Compute per-bucket DD-deduped mean OVL matrices.

    Each record must have keys:
        ``ovl`` (ndarray), ``rep_A``, ``rep_B``, ``view_A``, ``view_B``,
        ``client_A``, ``client_B``, ``view_pair_label``.

    Parameters
    ----------
    records : list of dicts
        Comparison records (cross-rep only; same-rep excluded upstream).
    system : DimerSystem
        Dimer topology.
    matrix_n : int
        Full matrix size (dimer_n + ligand_n).
    dd_atol : float
        Tolerance for DD block identity assertion.

    Returns
    -------
    means_per_bucket : dict[(view, clientA, clientB)] → ndarray
    eligible_unblocked : ndarray
        Simple mean over all records (diagnostic).
    audit : dict
        Per-bucket statistics and integrity checks.
    """
    dimer_n = system.dimer_n
    ligand_n = matrix_n - dimer_n
    dd_sl = slice(0, dimer_n)
    lg_sl = slice(dimer_n, matrix_n)

    # Bucket records by (view, client_A, client_B); cross-view excluded
    buckets: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    cross_view_records: List[Dict] = []
    for rec in records:
        if rec["view_A"] == rec["view_B"]:
            buckets[(rec["view_A"], rec["client_A"], rec["client_B"])].append(rec)
        else:
            cross_view_records.append(rec)

    means_per_bucket: Dict[Tuple[str, str, str], np.ndarray] = {}
    audit_per_bucket: Dict[str, Dict] = {}

    for bucket_key, bucket_recs in sorted(buckets.items()):
        view, cA, cB = bucket_key
        label = bucket_dataset_name(view, cA, cB)

        # Group by (rep_A, rep_B)
        rep_groups: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        for rec in bucket_recs:
            rep_groups[(rec["rep_A"], rec["rep_B"])].append(rec)

        sum_dd = np.zeros((dimer_n, dimer_n), dtype=np.float64)
        sum_dl = np.zeros((dimer_n, ligand_n), dtype=np.float64)
        sum_ll = np.zeros((ligand_n, ligand_n), dtype=np.float64)
        count_dd = 0
        count_dl = 0

        for (rep_A, rep_B), group in sorted(rep_groups.items()):
            # DD pick-first with integrity assertion
            dd_canonical = group[0]["ovl"][dd_sl, dd_sl]
            for other in group[1:]:
                dd_other = other["ovl"][dd_sl, dd_sl]
                if not np.allclose(dd_canonical, dd_other, rtol=0.0, atol=dd_atol):
                    max_dev = float(np.max(np.abs(dd_canonical - dd_other)))
                    raise ValueError(
                        f"DD block mismatch in bucket '{label}' "
                        f"rep-pair ({rep_A}, {rep_B}): max deviation "
                        f"{max_dev:.6e} exceeds atol={dd_atol}."
                    )
            sum_dd += dd_canonical
            count_dd += 1

            for rec in group:
                sum_dl += rec["ovl"][dd_sl, lg_sl]
                sum_ll += rec["ovl"][lg_sl, lg_sl]
                count_dl += 1

        if count_dd == 0:
            continue

        mean_dd = sum_dd / count_dd
        mean_dl = sum_dl / count_dl if count_dl > 0 else np.zeros_like(sum_dl)
        mean_ll = sum_ll / count_dl if count_dl > 0 else np.zeros_like(sum_ll)

        bucket_mean = np.zeros((matrix_n, matrix_n), dtype=np.float32)
        bucket_mean[dd_sl, dd_sl] = mean_dd
        bucket_mean[dd_sl, lg_sl] = mean_dl
        bucket_mean[lg_sl, dd_sl] = mean_dl.T
        bucket_mean[lg_sl, lg_sl] = mean_ll
        means_per_bucket[bucket_key] = bucket_mean

        audit_per_bucket[label] = {
            "view": view,
            "client_A": cA,
            "client_B": cB,
            "n_records": len(bucket_recs),
            "n_unique_rep_pairs": count_dd,
            "n_dl_ll_contributions": count_dl,
        }

    # Eligible unblocked mean (all records, no block awareness)
    if records:
        total = np.zeros((matrix_n, matrix_n), dtype=np.float64)
        for rec in records:
            total += rec["ovl"]
        eligible_unblocked = (total / len(records)).astype(np.float32)
    else:
        eligible_unblocked = np.zeros((matrix_n, matrix_n), dtype=np.float32)

    audit = {
        "dimer_n": dimer_n,
        "matrix_n": matrix_n,
        "n_buckets": len(means_per_bucket),
        "n_cross_view_excluded": len(cross_view_records),
        "total_records": len(records),
        "buckets": audit_per_bucket,
    }

    return means_per_bucket, eligible_unblocked, audit


def bucket_dataset_name(view: str, client_A: str, client_B: str) -> str:
    """H5 dataset name for a per-bucket mean.

    ``ovl_mean_deduped_PLmat1_BSN2`` or
    ``ovl_mean_deduped_native_BSN2_x_ICE1``.
    """
    if client_A == client_B:
        return f"ovl_mean_deduped_{view}_{client_A}"
    return f"ovl_mean_deduped_{view}_{client_A}_x_{client_B}"


def build_means_dict(
    means_per_bucket: Dict[Tuple[str, str, str], np.ndarray],
    eligible_unblocked: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Translate bucket keys to H5-ready dataset name → array dict."""
    out: Dict[str, np.ndarray] = {}
    for (view, cA, cB), M in means_per_bucket.items():
        out[bucket_dataset_name(view, cA, cB)] = M
    out["ovl_mean_eligible_unblocked"] = eligible_unblocked
    return out
