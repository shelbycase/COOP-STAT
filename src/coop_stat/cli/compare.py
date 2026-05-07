"""CLI: ``dimer-compare`` — statistical comparison of OVL matrices.

Loads cross and intra OVL H5 files produced by ``dimer-ovl``, then runs
KS-based pair selection (--emit-stats) and/or per-residue significance
weighting (--stat-sign).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from coop_stat.compare.extraction import (
    extract_values_for_pair,
    smart_mean,
)
from coop_stat.compare.output import (
    write_emit_stats_tsv,
    write_stat_sign_pdb,
    write_stat_sign_txt,
)
from coop_stat.compare.statistics import (
    compute_pair_stats,
    compute_stat_sign_weights,
    ks_pvalue,
)
from coop_stat.config import DimerSystem, lc8_system
from coop_stat.io.h5 import load_ovl_cross, load_ovl_intra
from coop_stat.compare.extraction import is_dimer_key
from coop_stat.tags import Tag, dd_sym_applies, parse_tag
from coop_stat.topology import (
    matched_binding_window,
    matched_client_for_comparison,
    window_roles,
)


# ── CLI ──────────────────────────────────────────────────────────────


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical comparison of OVL matrices."
    )
    p.add_argument("pdb1", help="Reference tag (e.g. BSN2s15)")
    p.add_argument("pdb2", help="Comparison tag")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ovl-dir", default="ovl_h5")
    p.add_argument("--outdir", default="plots")
    p.add_argument("--template-pdb", default=None,
                   help="PDB template for stat-sign output")

    # KS / filtering
    p.add_argument("--ks-method", choices=["auto", "exact", "asymp", "mc"],
                   default="auto")
    p.add_argument("--ks-mc-n", type=int, default=20000)
    p.add_argument("--ks-alpha", type=float, default=None,
                   help="KS p-value cutoff for emit-stats filtering")
    p.add_argument("--stat-sign-alpha", type=float, default=0.05)
    p.add_argument("--min-intra", type=float, default=None)
    p.add_argument("--min-intra-other", type=float, default=None)
    p.add_argument("--max-cross", type=float, default=None)
    p.add_argument("--pair-file", default=None,
                   help="Restrict to pairs listed in this file (i:j per line)")

    # Modes
    p.add_argument("--emit-stats", action="store_true")
    p.add_argument("--stat-sign", action="store_true")

    # System config
    p.add_argument("--system-preset", default="lc8",
                   choices=["lc8", "custom"])
    p.add_argument("--chain-length", type=int, default=None)
    p.add_argument("--top-dir", default=".")
    p.add_argument("--ligand-ranges", default="188-202,212-226")
    p.add_argument("--dd-only", action="store_true",
               help="Restrict to dimer-dimer pairs only (exclude DL/LL)")

    # Compat
    p.add_argument("--dimer-n", type=int, default=None)
    p.add_argument("--len1", type=int, default=8)
    p.add_argument("--len2", type=int, default=None)
    p.add_argument("--state1", default="s")
    p.add_argument("--state2", default=None)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


# ── OVL file loading ─────────────────────────────────────────────────


def _find_cross_h5(ref: str, cmp: str, ovl_dir: Path) -> Path:
    candidates = [
        ovl_dir / f"ovl_cross_{cmp}_vs_{ref}.h5",
        ovl_dir / f"ovl_cross_{ref}_vs_{cmp}.h5",
        ovl_dir / f"ovl_results_{cmp}_vs_{ref}_all_replicates.h5",
        ovl_dir / f"ovl_results_{ref}_vs_{cmp}_all_replicates.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No cross OVL H5 for {ref} vs {cmp}. Checked: {[str(c) for c in candidates]}"
    )


def _find_intra_h5(tag: str, ref: str, cmp: str, ovl_dir: Path) -> Optional[Path]:
    candidates = [
        ovl_dir / f"ovl_intra_{tag}_in_{cmp}_vs_{ref}.h5",
        ovl_dir / f"ovl_intra_{tag}_in_{ref}_vs_{cmp}.h5",
        ovl_dir / f"ovl_results_{tag}_vs_{tag}_all_replicates.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_bound_mean_h5(ref: str, cmp: str, ovl_dir: Path) -> Optional[Path]:
    candidates = [
        ovl_dir / f"ovl_boundstate_{cmp}_vs_{ref}.h5",
        ovl_dir / f"ovl_boundstate_{ref}_vs_{cmp}.h5",
        ovl_dir / f"ovl_results_{cmp}_vs_{ref}_all_replicates.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_mean_from_h5(path: Path, matched_client: Optional[str]) -> Optional[np.ndarray]:
    """Load per-bucket or fallback mean from an H5 that has mean datasets."""
    import h5py
    if path is None or not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        # Try per-bucket means matching the client
        if matched_client:
            for view in ("PLmat1", "PLmat2as1", "native"):
                k = f"ovl_mean_deduped_{view}_{matched_client}"
                if k in h5:
                    return h5[k][...]
        # Eligible unblocked
        if "ovl_mean_eligible_unblocked" in h5:
            return h5["ovl_mean_eligible_unblocked"][...]
        # Legacy
        for k in ("ovl_mean_deduped", "ovl_mean"):
            if k in h5:
                return h5[k][...]
    return None


def _read_pair_file(path: Optional[str]) -> Optional[Set[Tuple[int, int]]]:
    if not path:
        return None
    pairs = set()
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            tok = ln.split()[0]
            if ":" in tok:
                i, j = map(int, tok.split(":"))
                pairs.add((min(i, j), max(i, j)))
    return pairs


# ── Expanded pair labels (for TSV columns) ───────────────────────────


def _expanded_labels(ovl_dict, system, dd_sym, matched_client):
    """Get column labels for per-pair TSV values."""
    from coop_stat.compare.extraction import (
        is_dimer_key, is_intra_same_rep, client_match, rep_pair_from_key,
    )
    if ovl_dict is None:
        return []
    vks = sorted(
        k for k in ovl_dict
        if not is_dimer_key(k) and client_match(k, matched_client)
    )
    r2v = {}
    for vk in vks:
        r2v.setdefault(rep_pair_from_key(vk), []).append(vk)
    labels = []
    seen = set()
    for vk in vks:
        rp = rep_pair_from_key(vk)
        if rp in seen:
            continue
        seen.add(rp)
        fk = r2v[rp][0]
        if is_intra_same_rep(fk):
            for vi in r2v[rp]:
                labels.append(f"{vi}|diagnostic")
        else:
            for vi in r2v[rp]:
                labels.append(vi)
            if dd_sym:
                labels.append(f"{fk}|sym")
    return labels


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = _cli()

    # System config
    if args.system_preset == "lc8":
        system = lc8_system()
    elif args.chain_length:
        system = DimerSystem(chain_length=args.chain_length)
    else:
        sys.exit("ERROR: --chain-length required for custom systems")

    if args.dimer_n and args.dimer_n != system.dimer_n:
        system = DimerSystem(
            chain_length=args.dimer_n // 2,
            peptide_seqs=system.peptide_seqs,
            binding_windows_1=system.binding_windows_1,
            binding_windows_2=system.binding_windows_2,
        )

    ref_tag = parse_tag(args.pdb1)
    cmp_tag = parse_tag(args.pdb2)
    ref, cmp = str(ref_tag), str(cmp_tag)
    ovl_dir = Path(args.ovl_dir)
    od = Path(args.outdir)

    # Symmetry flags
    dsc = dd_sym_applies(ref_tag, cmp_tag)
    dsr = dd_sym_applies(ref_tag, ref_tag)
    dsm = dd_sym_applies(cmp_tag, cmp_tag)

    # Matched binding window / client
    mbw = matched_binding_window(ref_tag, cmp_tag, system)
    mc = matched_client_for_comparison(ref_tag, cmp_tag, mbw, system)
    wr = window_roles(ref_tag, cmp_tag, mbw, system, args.top_dir, args.ligand_ranges)

    print(f"[compare] {ref} vs {cmp}")
    print(f"  matched_bw={mbw}  matched_client={mc}")
    print(f"  dd_sym: cross={dsc} iref={dsr} icmp={dsm}")

    # Load OVL data
    # retired: to load keys from legacy codes
    #cross_path = _find_cross_h5(ref, cmp, ovl_dir)
    #cross = read_ovl_dict(cross_path)
    cross = load_ovl_cross(ref, cmp, ovl_dir)
    print(f"  cross: {len(cross)} keys from {cross_path.name}")

    mean_path = _find_bound_mean_h5(ref, cmp, ovl_dir)
    mean = _load_mean_from_h5(mean_path, mc)
    if mean is None:
        mean = smart_mean(cross, matched_client=mc)
    if mean is None:
        raise RuntimeError(f"No cross mean for {ref} vs {cmp}")
    
    # correction: makes the number of residues (dimeric or dimer+ligand optional)
    # nres = mean.shape[0]
    if args.dd_only:
        nres = system.dimer_n
        print(f"  nres={nres} (DD-only, dimer block)")
    else:
        nres = next(M.shape[0] for k, M in cross.items() if not is_dimer_key(k))
        print(f"  nres={nres} (full matrix, DD+DL+LL)")

    # retired: to load keys from legacy codes
    #iref_path = _find_intra_h5(ref, ref, cmp, ovl_dir)
    #iref = read_ovl_dict(iref_path) if iref_path else None
    
    try:
        iref = load_ovl_intra(ref, ref, cmp, ovl_dir)
    except FileNotFoundError:
        iref = None
    
    # retired: to load keys from legacy codes
    #icmp_path = _find_intra_h5(cmp, ref, cmp, ovl_dir)
    #icmp = read_ovl_dict(icmp_path) if icmp_path else None
    
    try:
        icmp = load_ovl_intra(cmp, ref, cmp, ovl_dir)
    except FileNotFoundError:
        icmp = None
    
    print(f"  iref: {len(iref) if iref else 0} keys")
    print(f"  icmp: {len(icmp) if icmp else 0} keys")

    if not args.emit_stats and not args.stat_sign:
        print("  ⚠️ Neither --emit-stats nor --stat-sign specified.")

    # ── Emit stats ───────────────────────────────────────────────────

    if args.emit_stats:
        rng = np.random.default_rng()
        pair_filter = _read_pair_file(args.pair_file)
        tr = args.min_intra
        to = args.min_intra_other if args.min_intra_other is not None else args.min_intra
        tc = args.max_cross
        ka = args.ks_alpha
        has_thresh = any(x is not None for x in (tr, to, tc))

        rows = []
        for i in range(nres):
            for j in range(i + 1, nres):
                if pair_filter and (i, j) not in pair_filter:
                    continue
                stats = compute_pair_stats(
                    i, j, cross, iref, icmp,
                    system=system, nres=nres,
                    dd_sym_cross=dsc, dd_sym_iref=dsr, dd_sym_icmp=dsm,
                    matched_client=mc, ks_method=args.ks_method,
                    ks_mc_n=args.ks_mc_n, rng=rng,
                )
                # Apply thresholds
                if has_thresh:
                    oc = tc is None or (stats["cm"] is not None and stats["cm"] <= tc)
                    orr = tr is None or (stats["rm"] is not None and stats["rm"] >= tr)
                    oo = to is None or (stats["mm"] is not None and stats["mm"] >= to)
                    hr, hm = stats["rm"] is not None, stats["mm"] is not None
                    io = (orr and oo) if (hr and hm) else (orr if hr else (oo if hm else True))
                    if not (oc and io):
                        continue
                if ka is not None and (stats["kc"] is None or stats["kc"] > ka):
                    continue
                stats["i"] = i
                stats["j"] = j
                rows.append(stats)

        # Sort: lowest cross mean first
        rows.sort(key=lambda r: (r["cm"] or float("inf"),
                                  -(r["rm"] or -1), -(r["mm"] or -1)))

        # Get column labels
        ck = _expanded_labels(cross, system, dsc, mc)
        rk = _expanded_labels(iref, system, dsr, mc) if iref else []
        mk = _expanded_labels(icmp, system, dsm, mc) if icmp else []

        sfn = f"{cmp}_vs_{ref}_{'highcrosslowintra' if has_thresh else 'allvals'}"
        tsv_path = od / f"{sfn}.tsv"
        write_emit_stats_tsv(rows, tsv_path, ck, rk, mk)
        print(f"  📄 {len(rows)} rows → {tsv_path}")

    # ── Stat-sign ────────────────────────────────────────────────────

    if args.stat_sign:
        weights = compute_stat_sign_weights(
            nres, cross, iref, icmp,
            system=system, ks_method=args.ks_method,
            ks_mc_n=args.ks_mc_n, alpha=args.stat_sign_alpha,
            dd_sym_cross=dsc, dd_sym_iref=dsr, dd_sym_icmp=dsm,
            matched_client=mc,
        )
        write_stat_sign_txt(weights, ref, cmp, od, system,
                            mbw=mbw, matched_client=mc, window_roles=wr)

        if args.template_pdb:
            tpl = Path(args.template_pdb)
            if tpl.exists():
                write_stat_sign_pdb(tpl, weights, ref, cmp, od, system)
            else:
                print(f"  ⚠️ template PDB not found: {tpl}")

        bi = set(system.binding_indices(bw=mbw))
        nz = sum(1 for i in bi if 0 <= i < len(weights) and weights[i] > 0)
        tw = float(sum(weights[i] for i in bi if 0 <= i < len(weights)))
        print(f"  🧪 stat_sign (bw={mbw}, client={mc}): {nz} nonzero, sum={tw:g}")

    # ── Log ──────────────────────────────────────────────────────────

    lp = Path(args.log_file) if args.log_file else od / f"{cmp}_vs_{ref}_input_files.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w") as fh:
        fh.write(f"# dimer-compare\n")
        fh.write(f"ref={ref} cmp={cmp} mc={mc} mbw={mbw}\n")
        fh.write(f"dd_sym={dsc},{dsr},{dsm}\n")
        fh.write(f"cross={cross_path}\n")
        if iref_path:
            fh.write(f"iref={iref_path}\n")
        if icmp_path:
            fh.write(f"icmp={icmp_path}\n")
    print(f"  📝 log → {lp}")


if __name__ == "__main__":
    main()
