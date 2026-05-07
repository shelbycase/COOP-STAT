"""
Statistical tests for OVL comparisons.
 
Core operation: for each residue pair (i, j), collect cross and intra OVL
samples, then apply two filters:
    1. KS test — are the cross and intra distributions significantly different?
    2. Direction check — is the intra mean greater than the cross mean?
 
A pair contributes to per-residue significance weights only if both filters pass.

"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import ks_2samp as _scipy_ks_2samp

from coop_stat.compare.extraction import extract_values_for_pair
from coop_stat.config import DimerSystem

from coop_stat.compare.extraction import extract_labeled_values_for_pair

# ── KS compatibility ─────────────────────────────────────────────────
# version check, dependency sanity check

def _detect_ks_method_support() -> bool:
    try:
        _scipy_ks_2samp([0.0, 1.0], [0.5, 1.5], alternative="two-sided", method="exact")
        return True
    except TypeError:
        return False


_HAS_METHOD = _detect_ks_method_support()
_WARNED = False


def ks_2samp(a, b, alternative="two-sided", method="auto"):
    """Scipy ks_2samp with fallback for older versions lacking ``method``."""
    global _WARNED
    if _HAS_METHOD:
        return _scipy_ks_2samp(a, b, alternative=alternative, method=method)
    if not _WARNED:
        warnings.warn(
            "ks_2samp(method=...) not supported; using default.", RuntimeWarning
        )
        _WARNED = True
    return _scipy_ks_2samp(a, b, alternative=alternative)


# ── P-value computation ──────────────────────────────────────────────


def ks_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "auto",
    mc_n: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[float]:
    """Compute two-sided KS p-value between samples *a* and *b*.

    Parameters
    ----------
    a, b : array-like
        Sample arrays (need ≥2 elements each).
    method : str
        ``"auto"``, ``"exact"``, ``"asymp"``, or ``"mc"`` (permutation).
    mc_n : int
        Number of permutations for MC method.
    rng : Generator, optional
        Random generator for MC method.

    Returns
    -------
    float or None
        P-value, or None if either sample has < 2 elements.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return None

    if method == "mc":
        if rng is None:
            rng = np.random.default_rng()
        n = a.size
        pooled = np.concatenate([a, b]).copy()
        D_obs = float(
            ks_2samp(a, b, alternative="two-sided", method="asymp").statistic
        )
        ge = 0
        for _ in range(mc_n):
            rng.shuffle(pooled)
            D_perm = float(
                ks_2samp(
                    pooled[:n], pooled[n:],
                    alternative="two-sided", method="asymp",
                ).statistic
            )
            if D_perm >= D_obs:
                ge += 1
        return float((ge + 1) / (mc_n + 1))

    meth = {"auto": "auto", "exact": "exact", "asymp": "asymp"}.get(method, "auto")
    return float(ks_2samp(a, b, alternative="two-sided", method=meth).pvalue)


# ── Stat-sign weights ────────────────────────────────────────────────


def compute_stat_sign_weights(
    nres: int,
    cross: Optional[Dict],
    iref: Optional[Dict],
    icmp: Optional[Dict],
    system: DimerSystem,
    ks_method: str = "auto",
    ks_mc_n: int = 20000,
    alpha: float = 0.05,
    dd_sym_cross: bool = False,
    dd_sym_iref: bool = False,
    dd_sym_icmp: bool = False,
    matched_client: Optional[str] = None,
) -> np.ndarray:
    """Compute per-residue statistical significance weights.

    For each pair (i, j):
        1. Extract cross, iref, icmp OVL samples.
        2. KS test cross vs each intra.
        3. If max(p-values) < alpha AND both intra means > cross mean,
           increment weights[i] and weights[j].

    Returns
    -------
    weights : ndarray of shape (nres,)
        Count of significant pairs each residue participates in.
    """
    weights = np.zeros(nres, dtype=float)
    if cross is None or (iref is None and icmp is None):
        return weights

    rng = np.random.default_rng()
    nprot = min(nres, system.dimer_n)
    if nprot < 2:
        return weights

    def _v(d, i, j, ds):
        return extract_values_for_pair(
            d, i, j, system=system, nres=nres,
            dd_sym=ds, matched_client=matched_client,
        ) if d else []

    for i in range(nprot):
        for j in range(i + 1, nprot):
            cv = _v(cross, i, j, dd_sym_cross)
            rv = _v(iref, i, j, dd_sym_iref)
            mv = _v(icmp, i, j, dd_sym_icmp)

            if len(cv) < 2 or (len(rv) < 2 and len(mv) < 2):
                continue

            pr = ks_pvalue(cv, rv, method=ks_method, mc_n=ks_mc_n, rng=rng) if len(rv) >= 2 else None
            pm = ks_pvalue(cv, mv, method=ks_method, mc_n=ks_mc_n, rng=rng) if len(mv) >= 2 else None
            pcs = [p for p in (pr, pm) if p is not None]
            if not pcs or max(pcs) >= alpha:
                continue

            cm = float(np.mean(cv))
            rm = float(np.mean(rv)) if rv else np.nan
            mm = float(np.mean(mv)) if mv else np.nan

            if any(np.isnan(x) for x in (cm, rm, mm)):
                continue
            if not (rm > cm and mm > cm):
                continue

            weights[i] += 1.0
            weights[j] += 1.0

    return weights


# ── Emit-stats row computation ───────────────────────────────────────


def compute_pair_stats(
    i: int,
    j: int,
    cross: Optional[Dict],
    iref: Optional[Dict],
    icmp: Optional[Dict],
    system: DimerSystem,
    nres: int,
    dd_sym_cross: bool,
    dd_sym_iref: bool,
    dd_sym_icmp: bool,
    matched_client: Optional[str],
    ks_method: str = "auto",
    ks_mc_n: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Compute summary statistics for a single pair (i, j).

    Returns dict with keys:
        cm, cs, rm, rs, mm, ms (means and SDs),
        kr, km, kc (KS p-values).
    """
    def _v(d, ds):
        return extract_values_for_pair(
            d, i, j, system=system, nres=nres,
            dd_sym=ds, matched_client=matched_client,
        ) if d else []

    cv = _v(cross, dd_sym_cross)
    rv = _v(iref, dd_sym_iref)
    mv = _v(icmp, dd_sym_icmp)

    def _ms(vals):
        if not vals:
            return None, None
        a = np.asarray(vals, float)
        return float(a.mean()), float(a.std(ddof=0))

    cm, cs = _ms(cv)
    rm, rs = _ms(rv)
    mm, ms = _ms(mv)


    kr = ks_pvalue(cv, rv, method=ks_method, mc_n=ks_mc_n, rng=rng) if rv else None
    km = ks_pvalue(cv, mv, method=ks_method, mc_n=ks_mc_n, rng=rng) if mv else None
    kl = [p for p in (kr, km) if p is not None]
    kc = max(kl) if kl else None

    cross_labeled = dict(extract_labeled_values_for_pair(
        cross, i, j, system=system, nres=nres,
        dd_sym=dd_sym_cross, matched_client=matched_client,
    )) if cross else {}
    iref_labeled = dict(extract_labeled_values_for_pair(
        iref, i, j, system=system, nres=nres,
        dd_sym=dd_sym_iref, matched_client=matched_client,
    )) if iref else {}
    icmp_labeled = dict(extract_labeled_values_for_pair(
        icmp, i, j, system=system, nres=nres,
        dd_sym=dd_sym_icmp, matched_client=matched_client,
    )) if icmp else {}

    return {"cm": cm, "cs": cs, "rm": rm, "rs": rs, "mm": mm, "ms": ms,
            "kr": kr, "km": km, "kc": kc,
            "cross_vals": cross_labeled,
            "iref_vals": iref_labeled,
            "icmp_vals": icmp_labeled}
